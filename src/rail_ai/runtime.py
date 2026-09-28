"""Hardware runtime detection and controlled CPU fallback."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig


@dataclass
class RuntimeState:
    requested_device: str
    effective_device: str
    provider: str
    platform: str
    gpu_available: bool
    accelerator: str = ""
    gpu_pci_address: str = ""
    vaapi_device: str = ""
    provider_fallback_reason: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeManager:
    _onnx_patched = False

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = RuntimeState(
            requested_device=config.requested_device,
            effective_device="cpu",
            provider="CPUExecutionProvider" if config.model_format == "onnx" else "pytorch-cpu",
            platform=config.platform,
            gpu_available=False,
        )

    @staticmethod
    def _import_torch():
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed in this image") from exc
        return torch

    @staticmethod
    def _onnx_providers() -> list[str]:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return []
        return list(ort.get_available_providers())

    def _warmup_gpu(self, torch: Any) -> None:
        tensor = torch.ones((64, 64), device="cuda:0", dtype=torch.float32)
        result = (tensor @ tensor).sum()
        _ = float(result.cpu().item())
        torch.cuda.synchronize()

    def _gpu_kind(self, torch: Any) -> str:
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"

    def select(self) -> RuntimeState:
        requested = self.config.requested_device
        if requested == "cpu":
            return self.force_cpu("")

        try:
            torch = self._import_torch()
            if not bool(torch.cuda.is_available()):
                raise RuntimeError("PyTorch reports no available GPU")

            gpu_kind = self._gpu_kind(torch)
            if requested == "rocm:0" and gpu_kind != "rocm":
                raise RuntimeError("ROCm was requested but this PyTorch build is not ROCm-enabled")
            if requested == "cuda:0" and gpu_kind != "cuda":
                raise RuntimeError("CUDA was requested but this PyTorch build is ROCm-enabled")
            if gpu_kind == "rocm" and (not Path("/dev/kfd").exists() or not Path("/dev/dri").exists()):
                raise RuntimeError("ROCm device nodes /dev/kfd and /dev/dri are not available")

            self._warmup_gpu(torch)
            provider = self._provider_for_gpu(gpu_kind)
            pci_address = self._gpu_pci_address(torch)
            self.state = RuntimeState(
                requested_device=requested,
                effective_device=f"{gpu_kind}:0",
                provider=provider,
                platform=self.config.platform if self.config.platform != "auto" else gpu_kind,
                gpu_available=True,
                accelerator=self._accelerator_name(torch, gpu_kind),
                gpu_pci_address=pci_address,
                vaapi_device=self._vaapi_device_for_pci(pci_address) if gpu_kind == "rocm" else "",
                provider_fallback_reason="",
            )
            self._configure_onnx_provider()
            return self.state
        except Exception as exc:
            if self.config.device_fallback != "cpu":
                raise
            return self.force_cpu(str(exc))

    def _provider_for_gpu(self, gpu_kind: str) -> str:
        if self.config.model_format == "tensorrt":
            if gpu_kind != "cuda":
                raise RuntimeError("TensorRT requires an NVIDIA CUDA GPU")
            return "TensorRT"
        if self.config.model_format != "onnx":
            return "pytorch-rocm" if gpu_kind == "rocm" else "pytorch-cuda"

        providers = self._onnx_providers()
        wanted = "MIGraphXExecutionProvider" if gpu_kind == "rocm" else "CUDAExecutionProvider"
        if wanted not in providers:
            raise RuntimeError(f"ONNX provider {wanted} is unavailable; available providers: {providers}")
        os.environ["ORT_PROVIDER"] = wanted
        return wanted

    @staticmethod
    def _accelerator_name(torch: Any, gpu_kind: str) -> str:
        name = str(torch.cuda.get_device_name(0))
        if gpu_kind == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability(0)
                return f"{name} sm_{major}{minor}"
            except Exception:
                pass
        return name

    @staticmethod
    def _gpu_pci_address(torch: Any) -> str:
        try:
            properties = torch.cuda.get_device_properties(0)
            return (
                f"{int(properties.pci_domain_id):04x}:"
                f"{int(properties.pci_bus_id):02x}:"
                f"{int(properties.pci_device_id):02x}.0"
            )
        except Exception:
            return ""

    @staticmethod
    def _vaapi_device_for_pci(pci_address: str) -> str:
        if not pci_address:
            return ""
        drm_root = Path("/sys/class/drm")
        if not drm_root.exists():
            return ""
        for render_node in sorted(drm_root.glob("renderD*")):
            try:
                device_path = render_node.joinpath("device").resolve()
            except OSError:
                continue
            if device_path.name.lower() == pci_address.lower():
                candidate = Path("/dev/dri") / render_node.name
                if candidate.exists():
                    return str(candidate)
        return ""

    def force_cpu(self, reason: str) -> RuntimeState:
        os.environ["ORT_PROVIDER"] = "CPUExecutionProvider"
        self.state = RuntimeState(
            requested_device=self.config.requested_device,
            effective_device="cpu",
            provider="CPUExecutionProvider" if self.config.model_format == "onnx" else "pytorch-cpu",
            platform=self.config.platform,
            gpu_available=False,
            fallback_used=bool(reason),
            fallback_reason=reason,
        )
        self._configure_onnx_provider()
        return self.state

    def _configure_onnx_provider(self) -> None:
        if self.config.model_format != "onnx" or RuntimeManager._onnx_patched:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return

        original_session = ort.InferenceSession
        setattr(ort, "_rail_ai_original_inference_session", original_session)

        def provider_session(*args: Any, **kwargs: Any):
            wanted = os.getenv("ORT_PROVIDER", "CPUExecutionProvider")
            available = list(ort.get_available_providers())
            if wanted not in available:
                raise RuntimeError(f"Requested ONNX provider {wanted} is unavailable; available providers: {available}")
            if wanted == "MIGraphXExecutionProvider":
                model_path = Path(args[0] if args else kwargs.get("path_or_bytes", ""))
                cache_path = model_path.parent / "migraphx-cache" / model_path.stem
                cache_path.mkdir(parents=True, exist_ok=True)
                options = {
                    "device_id": "0",
                    "migraphx_model_cache_dir": str(cache_path),
                }
                if self.config.model_precision == "fp16":
                    options["migraphx_fp16_enable"] = "1"
                elif self.config.model_precision == "int8":
                    calibration_table = model_path.with_suffix(".calibration.flatbuffers")
                    if not calibration_table.is_file():
                        raise RuntimeError(f"Missing MIGraphX INT8 calibration table: {calibration_table}")
                    options.update(
                        {
                            "migraphx_int8_enable": "1",
                            "migraphx_int8_calibration_table_name": str(calibration_table),
                            "migraphx_int8_use_native_calibration_table": "0",
                        }
                    )
                kwargs["providers"] = [(wanted, options), "CPUExecutionProvider"]
            else:
                kwargs["providers"] = [wanted] if wanted == "CPUExecutionProvider" else [wanted, "CPUExecutionProvider"]
            session = original_session(*args, **kwargs)
            active_providers = list(session.get_providers())
            if wanted != "CPUExecutionProvider" and wanted not in active_providers:
                raise RuntimeError(
                    f"Requested ONNX provider {wanted} fell back unexpectedly; "
                    f"active providers: {active_providers}"
                )
            return session

        ort.InferenceSession = provider_session
        RuntimeManager._onnx_patched = True

    @property
    def ultralytics_device(self) -> str:
        return "cpu" if self.state.effective_device == "cpu" else "cuda:0"

    def can_runtime_fallback(self) -> bool:
        return self.state.effective_device != "cpu" and not self.state.fallback_used

    def gpu_utilization_percent(self) -> float | None:
        if not self.state.gpu_available:
            return None
        if self.state.effective_device.startswith("rocm") and self.state.vaapi_device:
            utilization = (
                Path("/sys/class/drm")
                / Path(self.state.vaapi_device).name
                / "device"
                / "gpu_busy_percent"
            )
            try:
                return float(utilization.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                return None
        if self.state.effective_device.startswith("cuda"):
            try:
                torch = self._import_torch()
                return float(torch.cuda.utilization(0))
            except Exception:
                return None
        return None
