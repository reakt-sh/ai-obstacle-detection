"""Unified model artifact caching, export and loading."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .runtime import RuntimeManager


TENSORRT_EXPORT_POLICY = "builder-level5-int8-segmentation-proto-fp16-v2"
MIGRAPHX_ONNX_OPSET = 17
MIGRAPHX_EXPORT_POLICY = f"opset-{MIGRAPHX_ONNX_OPSET}-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_precision(config: AppConfig, runtime: RuntimeManager) -> str:
    if runtime.state.effective_device == "cpu" and config.model_precision == "int8":
        raise RuntimeError("INT8 requires an active GPU runtime; CPU fallback is disabled")
    if runtime.state.effective_device == "cpu" and config.model_precision == "fp16":
        return "fp32"
    return config.model_precision


def runtime_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "ultralytics", "onnxruntime", "tensorrt"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def calibration_image_paths(config: AppConfig) -> list[Path]:
    calibration_yaml = config.int8_data
    if calibration_yaml is None or not calibration_yaml.is_file():
        raise FileNotFoundError(f"INT8 calibration YAML not found: {calibration_yaml}")

    import yaml  # type: ignore

    data = yaml.safe_load(calibration_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data.get("path") or calibration_yaml.parent)
    if not dataset_root.is_absolute():
        dataset_root = (calibration_yaml.parent / dataset_root).resolve()
    sources = data.get("train") or data.get("val")
    if not sources:
        raise ValueError(f"INT8 calibration YAML has no train or val source: {calibration_yaml}")
    if isinstance(sources, str):
        sources = [sources]

    images: list[Path] = []
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    for source in sources:
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = dataset_root / source_path
        if source_path.is_dir():
            images.extend(path for path in source_path.rglob("*") if path.suffix.lower() in extensions)
        elif source_path.is_file() and source_path.suffix.lower() == ".txt":
            for line in source_path.read_text(encoding="utf-8").splitlines():
                image_path = Path(line.strip())
                if not image_path.is_absolute():
                    image_path = source_path.parent / image_path
                if image_path.is_file() and image_path.suffix.lower() in extensions:
                    images.append(image_path.resolve())
        elif source_path.is_file() and source_path.suffix.lower() in extensions:
            images.append(source_path.resolve())

    images = sorted(set(images))
    if not images:
        raise ValueError(f"No calibration images found through {calibration_yaml}")
    limit = min(config.int8_calibration_samples, len(images))
    if limit == len(images):
        return images
    if limit == 1:
        return [images[len(images) // 2]]
    return [images[round(index * (len(images) - 1) / (limit - 1))] for index in range(limit)]


def calibration_fingerprint(config: AppConfig) -> str:
    if config.model_precision != "int8":
        return ""
    calibration_yaml = config.int8_data
    assert calibration_yaml is not None
    digest = hashlib.sha256()
    digest.update(str(calibration_yaml.resolve()).encode("utf-8"))
    digest.update(calibration_yaml.read_bytes())
    for image in calibration_image_paths(config):
        stat = image.stat()
        digest.update(str(image).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def cache_key(config: AppConfig, runtime: RuntimeManager, precision: str) -> str:
    data = {
        "model": file_sha256(config.model),
        "format": config.model_format,
        "precision": precision,
        "imgsz": config.imgsz,
        "platform": runtime.state.platform,
        "provider": runtime.state.provider,
        "accelerator": runtime.state.accelerator,
        "runtime_versions": runtime_versions(),
        "migraphx_export_policy": MIGRAPHX_EXPORT_POLICY if config.model_format == "onnx" else "",
        "tensorrt_export_policy": TENSORRT_EXPORT_POLICY if config.model_format == "tensorrt" else "",
        "int8_calibration": calibration_fingerprint(config),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _model_class():
    from ultralytics import YOLO  # type: ignore

    return YOLO


def _write_calibration_yaml(config: AppConfig, cache_dir: Path, model: Any) -> Path:
    import yaml  # type: ignore

    images = calibration_image_paths(config)
    image_list = cache_dir / "model-calibration-images.txt"
    image_list.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    names = getattr(model, "names", None) or {0: "object", 1: "rail"}
    dataset = {
        "path": "/",
        "train": str(image_list),
        "val": str(image_list),
        "names": names,
    }
    destination = cache_dir / "model-calibration.yaml"
    destination.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return destination


def _letterbox_image(path: Path, imgsz: int, cv2: Any, np: Any) -> Any:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unable to read calibration image: {path}")
    height, width = image.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    left = (imgsz - resized_width) // 2
    top = (imgsz - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0


def _create_migraphx_calibration_table(onnx_path: Path, table_path: Path, config: AppConfig) -> None:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table  # type: ignore

    original_session = getattr(ort, "_rail_ai_original_inference_session", ort.InferenceSession)
    input_names = [
        item.name
        for item in original_session(str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()
    ]

    class ImageReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._items = iter(calibration_image_paths(config))

        def get_next(self):
            try:
                path = next(self._items)
            except StopIteration:
                return None
            return {input_names[0]: _letterbox_image(path, config.imgsz, cv2, np)}

    with tempfile.TemporaryDirectory(prefix="migraphx-calibration-", dir=table_path.parent) as temp_dir:
        temp = Path(temp_dir)
        patched_session = ort.InferenceSession
        ort.InferenceSession = original_session
        try:
            calibrator = create_calibrator(
                str(onnx_path),
                augmented_model_path=str(temp / "augmented.onnx"),
                providers=["CPUExecutionProvider"],
            )
            calibrator.collect_data(ImageReader())
            ranges = calibrator.compute_data()
            write_calibration_table(ranges, dir=str(temp))
        finally:
            ort.InferenceSession = patched_session
        shutil.move(str(temp / "calibration.flatbuffers"), table_path)


def _export_model(
    source: Path,
    destination: Path,
    config: AppConfig,
    runtime: RuntimeManager,
    precision: str,
) -> Path:
    work_source = destination.parent / "model.pt"
    shutil.copy2(source, work_source)
    model = _model_class()(str(work_source), task="segment")
    export_format = "onnx" if config.model_format == "onnx" else "engine"
    kwargs: dict[str, Any] = {
        "format": export_format,
        "imgsz": config.imgsz,
        "batch": 1,
        "device": runtime.ultralytics_device,
        "simplify": False,
    }
    if precision == "fp16" and export_format == "engine":
        kwargs["half"] = True
    if export_format == "onnx" and config.platform == "amd":
        # MIGraphX 7.2 cannot parse the Resize.keep_aspect_ratio_policy
        # attribute emitted with opset 18.
        kwargs["opset"] = MIGRAPHX_ONNX_OPSET
    if precision == "int8" and export_format == "engine":
        kwargs["int8"] = True
        kwargs["data"] = str(_write_calibration_yaml(config, destination.parent, model))
    if export_format == "engine":
        kwargs["workspace"] = config.export_workspace_gib

    exported = Path(model.export(**kwargs))
    if exported.is_dir():
        raise RuntimeError(f"Unexpected directory export for {config.model_format}: {exported}")
    shutil.move(str(exported), destination)
    if config.platform == "amd" and config.model_format == "onnx" and precision == "int8":
        _create_migraphx_calibration_table(destination, destination.with_suffix(".calibration.flatbuffers"), config)
    work_source.unlink(missing_ok=True)
    return destination


def resolve_model_path(config: AppConfig, runtime: RuntimeManager) -> tuple[Path, str]:
    if not config.model.exists():
        raise FileNotFoundError(f"Model file not found: {config.model}")
    if config.model_precision == "int8":
        calibration_image_paths(config)

    precision = effective_precision(config, runtime)
    if config.model_format == "pytorch":
        return config.model, precision
    if config.model_format == "tensorrt" and runtime.state.effective_device == "cpu":
        raise RuntimeError("TensorRT cannot run after a CPU fallback")

    suffix = ".onnx" if config.model_format == "onnx" else ".engine"
    key = cache_key(config, runtime, precision)
    cache_dir = config.export_dir / key
    model_path = cache_dir / f"model-{precision}{suffix}"
    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    lock_path = config.export_dir / f"{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not model_path.exists():
            _export_model(config.model, model_path, config, runtime, precision)
        if config.platform == "amd" and config.model_format == "onnx" and precision == "int8":
            table_path = model_path.with_suffix(".calibration.flatbuffers")
            if not table_path.exists():
                _create_migraphx_calibration_table(model_path, table_path, config)
        manifest_path.write_text(
            json.dumps(
                {
                    "cache_key": key,
                    "format": config.model_format,
                    "precision": precision,
                    "platform": runtime.state.platform,
                    "provider": runtime.state.provider,
                    "accelerator": runtime.state.accelerator,
                    "runtime_versions": runtime_versions(),
                    "migraphx_export_policy": MIGRAPHX_EXPORT_POLICY if config.model_format == "onnx" else "",
                    "tensorrt_export_policy": TENSORRT_EXPORT_POLICY,
                    "model_source_sha256": file_sha256(config.model),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return model_path, precision


@dataclass
class UnifiedModel:
    model: Any
    path: Path
    precision: str
    format: str

    @property
    def effective_precision(self) -> str:
        return self.precision

    @property
    def supports_staged_inference(self) -> bool:
        """Return whether the warmed model supports the staged TensorRT path."""
        return self.format == "tensorrt" and getattr(self.model, "predictor", None) is not None

    def preprocess(self, frame: Any) -> Any:
        if not self.supports_staged_inference:
            raise RuntimeError("Staged preprocessing requires a warmed TensorRT predictor")
        return self.model.predictor.preprocess([frame])

    def predict_preprocessed(self, frame: Any, image: Any) -> Any:
        """Run the GPU stage and return a binding-independent CPU snapshot.

        TensorRT reuses its output bindings on every invocation. Moving the
        completed result to host memory before the next invocation prevents a
        following frame from overwriting boxes and masks still in use by the
        post-processing stage.
        """
        if not self.supports_staged_inference:
            raise RuntimeError("Staged inference requires a warmed TensorRT predictor")
        predictor = self.model.predictor
        raw = predictor.inference(image)
        results = predictor.postprocess(raw, image, [frame])
        return [result.cpu() for result in results]

    def predict(self, frame: Any, config: AppConfig, runtime: RuntimeManager) -> Any:
        # One predictor must retain candidates for both post-processing
        # thresholds. Rail and object confidence are applied separately later.
        confidence = min(config.rail_confidence, config.object_confidence)
        device = "cpu" if self.format == "onnx" else runtime.ultralytics_device
        return self.model.predict(
            frame,
            imgsz=config.imgsz,
            device=device,
            conf=confidence,
            half=self.precision == "fp16" and self.format == "pytorch",
            verbose=False,
        )


def load_model(config: AppConfig, runtime: RuntimeManager) -> UnifiedModel:
    model_path, precision = resolve_model_path(config, runtime)
    model = _model_class()(str(model_path), task="segment")
    names = getattr(model, "names", {})
    if "rail" not in set(names.values() if isinstance(names, dict) else names):
        raise ValueError(f"Unified model must contain a 'rail' class: {config.model}")
    loaded = UnifiedModel(model, model_path, precision, config.model_format)

    import numpy as np  # type: ignore

    warmup = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
    loaded.predict(warmup, config, runtime)
    return loaded
