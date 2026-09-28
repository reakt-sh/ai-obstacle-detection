"""Application configuration and validation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


FORMATS = ("pytorch", "onnx", "tensorrt")
PRECISIONS = ("fp32", "fp16", "int8")
DEVICES = ("auto", "cpu", "rocm:0", "cuda:0")
OUTPUT_ENCODERS = ("auto", "software", "vaapi", "nvenc", "nvv4l2")
INPUT_DECODERS = ("auto", "software", "vaapi", "cuda", "nvv4l2")
PLATFORMS = ("auto", "cpu", "amd", "rtx", "jetson")


class BooleanOptionalAction(argparse.Action):
    """Python 3.8 compatible equivalent of argparse.BooleanOptionalAction."""

    def __init__(self, option_strings, dest, default=None, **kwargs):
        expanded = []
        for option in option_strings:
            expanded.append(option)
            if option.startswith("--"):
                expanded.append(f"--no-{option[2:]}")
        super().__init__(option_strings=expanded, dest=dest, nargs=0, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, values
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def normalize_format(value: str) -> str:
    aliases = {
        "pt": "pytorch",
        "torch": "pytorch",
        "trt": "tensorrt",
        "engine": "tensorrt",
    }
    return aliases.get(value.lower(), value.lower())


def normalize_device(value: str) -> str:
    aliases = {"cuda": "cuda:0", "rocm": "rocm:0", "gpu": "auto"}
    return aliases.get(value.lower(), value.lower())


def normalize_encoder(value: str) -> str:
    aliases = {
        "x264enc": "software",
        "libx264": "software",
        "h264_nvenc": "nvenc",
        "nvv4l2h264enc": "nvv4l2",
    }
    return aliases.get(value.lower(), value.lower())


@dataclass
class AppConfig:
    input_url: str
    input_protocol: str = "rtsp"
    model_format: str = "pytorch"
    model_precision: str = "fp32"
    requested_device: str = "auto"
    device_fallback: str = "cpu"
    platform: str = "auto"
    model: Path = Path("model/reaktrailai-yolo26m-seg.pt")
    export_dir: Path = Path("/models/exports")
    int8_data: Path | None = None
    imgsz: int = 640
    webrtc_enabled: bool = True
    webrtc_rtp_url: str = "rtp://127.0.0.1:5004?pkt_size=1200"
    webrtc_encoder: str = "auto"
    vaapi_device: str = "auto"
    h264_bitrate_kbps: int = 30000
    output_fps: float = 25.0
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    max_frames: int = 0
    latency_csv: Path | None = None
    status_json: Path | None = None
    record_dir: Path | None = None
    record_annotated: bool = False
    debug_display: bool = False
    input_decoder: str = "auto"
    export_workspace_gib: float = 4.0
    int8_calibration_samples: int = 256
    rail_confidence: float = 0.25
    object_confidence: float = 0.15
    object_hold_seconds: float = 1.0

    def validate(self) -> None:
        self.model_format = normalize_format(self.model_format)
        self.requested_device = normalize_device(self.requested_device)
        self.webrtc_encoder = normalize_encoder(self.webrtc_encoder)
        self.input_protocol = self.input_protocol.lower()
        self.model_precision = self.model_precision.lower()
        self.platform = self.platform.lower()
        self.model = Path(self.model)
        self.export_dir = Path(self.export_dir)
        if self.int8_data is not None:
            self.int8_data = Path(self.int8_data)

        if not self.input_url:
            raise ValueError("INPUT_URL is required")
        if not self.input_url.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("INPUT_URL must be an rtsp:// or rtsps:// URL")
        if self.input_protocol != "rtsp":
            raise ValueError("Only INPUT_PROTOCOL=rtsp is supported")
        if self.model_format not in FORMATS:
            raise ValueError(f"MODEL_FORMAT must be one of {FORMATS}")
        if self.model_precision not in PRECISIONS:
            raise ValueError(f"MODEL_PRECISION must be one of {PRECISIONS}")
        if self.requested_device not in DEVICES:
            raise ValueError(f"DEVICE must be one of {DEVICES}")
        if self.device_fallback != "cpu":
            raise ValueError("Only DEVICE_FALLBACK=cpu is currently supported")
        if self.webrtc_encoder not in OUTPUT_ENCODERS:
            raise ValueError(f"WEBRTC_ENCODER must be one of {OUTPUT_ENCODERS}")
        if self.input_decoder not in INPUT_DECODERS:
            raise ValueError(f"INPUT_DECODER must be one of {INPUT_DECODERS}")
        if self.platform not in PLATFORMS:
            raise ValueError(f"RAIL_AI_PLATFORM must be one of {PLATFORMS}")
        if self.imgsz < 32:
            raise ValueError("IMGSZ must be >= 32")
        if not 0.0 <= self.rail_confidence <= 1.0:
            raise ValueError("RAIL_CONFIDENCE must be in the range 0.0..1.0")
        if not 0.0 <= self.object_confidence <= 1.0:
            raise ValueError("OBJECT_CONFIDENCE must be in the range 0.0..1.0")
        if not 0.0 <= self.object_hold_seconds <= 60.0:
            raise ValueError("OBJECT_HOLD_SECONDS must be in the range 0.0..60.0")
        if self.output_fps <= 0:
            raise ValueError("OUTPUT_FPS must be > 0")
        if self.h264_bitrate_kbps <= 0:
            raise ValueError("H264_BITRATE_KBPS must be > 0")
        if self.max_frames < 0:
            raise ValueError("MAX_FRAMES must be >= 0")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("API_PORT must be in the range 1..65535")
        if self.export_workspace_gib <= 0:
            raise ValueError("TRT_WORKSPACE must be > 0")
        if self.record_annotated and self.record_dir is None:
            raise ValueError("RECORD_ANNOTATED requires RECORD_DIR")
        if self.model_precision == "int8" and self.int8_data is None:
            raise ValueError(
                "INT8 requires representative calibration data. Set INT8_DATA for the unified model."
            )
        if self.int8_calibration_samples < 1:
            raise ValueError("INT8_CALIBRATION_SAMPLES must be >= 1")
        if self.model_format == "tensorrt" and self.platform in {"amd", "cpu"}:
            raise ValueError("TensorRT is supported only on NVIDIA RTX and Jetson platforms")
        if self.model_precision == "int8":
            if self.platform == "amd" and self.model_format != "onnx":
                raise ValueError("AMD INT8 requires MODEL_FORMAT=onnx")
            if self.platform in {"rtx", "jetson"} and self.model_format != "tensorrt":
                raise ValueError("NVIDIA INT8 requires MODEL_FORMAT=tensorrt")
            if self.platform == "cpu":
                raise ValueError("INT8 GPU export is not supported by the CPU profile")
        supported_modes = {
            "cpu": {("pytorch", "fp32")},
            "amd": {
                ("pytorch", "fp32"),
                ("pytorch", "fp16"),
                ("onnx", "fp16"),
                ("onnx", "int8"),
            },
            "rtx": {
                ("pytorch", "fp32"),
                ("pytorch", "fp16"),
                ("tensorrt", "fp16"),
                ("tensorrt", "int8"),
            },
            "jetson": {
                ("pytorch", "fp32"),
                ("pytorch", "fp16"),
                ("tensorrt", "fp16"),
                ("tensorrt", "int8"),
            },
        }
        if self.platform in supported_modes and (self.model_format, self.model_precision) not in supported_modes[self.platform]:
            combinations = ", ".join(
                f"{model_format}/{precision}"
                for model_format, precision in sorted(supported_modes[self.platform])
            )
            raise ValueError(
                f"Unsupported model mode {self.model_format}/{self.model_precision} "
                f"for platform {self.platform}; supported: {combinations}"
            )
        if self.model_precision == "fp16" and self.requested_device == "cpu":
            raise ValueError("FP16 inference is not supported by the CPU profile")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-latency rail and obstacle inference pipeline")
    parser.add_argument("--source", dest="input_url", default=os.getenv("INPUT_URL"))
    parser.add_argument("--source-type", dest="input_protocol", choices=("rtsp",), default="rtsp")
    parser.add_argument("--model-format", default=os.getenv("MODEL_FORMAT", "pytorch"))
    parser.add_argument("--model-precision", default=os.getenv("MODEL_PRECISION", "fp32"))
    parser.add_argument("--device", dest="requested_device", default=os.getenv("DEVICE", "auto"))
    parser.add_argument("--device-fallback", default=os.getenv("DEVICE_FALLBACK", "cpu"))
    parser.add_argument("--platform", default=os.getenv("RAIL_AI_PLATFORM", "auto"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL", "model/reaktrailai-yolo26m-seg.pt")),
    )
    parser.add_argument("--export-dir", type=Path, default=Path(os.getenv("EXPORT_DIR", "/models/exports")))
    parser.add_argument("--int8-data", type=Path, default=Path(os.environ["INT8_DATA"]) if os.getenv("INT8_DATA") else None)
    parser.add_argument("--imgsz", type=int, default=env_int("IMGSZ", 640))
    parser.add_argument("--webrtc", dest="webrtc_enabled", action=BooleanOptionalAction, default=env_bool("WEBRTC_ENABLED", True))
    parser.add_argument("--webrtc-rtp-url", default=os.getenv("WEBRTC_RTP_URL", "rtp://127.0.0.1:5004?pkt_size=1200"))
    parser.add_argument("--webrtc-encoder", default=os.getenv("WEBRTC_ENCODER", "auto"))
    parser.add_argument("--vaapi-device", default=os.getenv("VAAPI_DEVICE", "auto"))
    parser.add_argument("--h264-bitrate-kbps", type=int, default=env_int("H264_BITRATE_KBPS", 30000))
    parser.add_argument("--output-fps", type=float, default=env_float("OUTPUT_FPS", 25.0))
    parser.add_argument("--serve", dest="api_enabled", action=BooleanOptionalAction, default=env_bool("API_ENABLED", True))
    parser.add_argument("--host", dest="api_host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", dest="api_port", type=int, default=env_int("API_PORT", 5000))
    parser.add_argument("--max-frames", type=int, default=env_int("MAX_FRAMES", 0))
    parser.add_argument("--latency-csv", type=Path, default=Path(os.environ["LATENCY_CSV"]) if os.getenv("LATENCY_CSV") else None)
    parser.add_argument("--status-json", type=Path, default=Path(os.environ["STATUS_JSON"]) if os.getenv("STATUS_JSON") else None)
    parser.add_argument("--record-dir", type=Path, default=Path(os.environ["RECORD_DIR"]) if os.getenv("RECORD_DIR") else None)
    parser.add_argument("--record-annotated", action=BooleanOptionalAction, default=env_bool("RECORD_ANNOTATED", False))
    parser.add_argument("--debug-display", action=BooleanOptionalAction, default=env_bool("DEBUG_DISPLAY", False))
    parser.add_argument("--input-decoder", choices=INPUT_DECODERS, default=os.getenv("INPUT_DECODER", "auto"))
    parser.add_argument("--export-workspace-gib", type=float, default=env_float("TRT_WORKSPACE", 4.0))
    parser.add_argument("--int8-calibration-samples", type=int, default=env_int("INT8_CALIBRATION_SAMPLES", 256))
    parser.add_argument("--rail-confidence", type=float, default=env_float("RAIL_CONFIDENCE", 0.25))
    parser.add_argument("--object-confidence", type=float, default=env_float("OBJECT_CONFIDENCE", 0.15))
    parser.add_argument("--object-hold-seconds", type=float, default=env_float("OBJECT_HOLD_SECONDS", 1.0))
    return parser


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    config = AppConfig(**vars(build_parser().parse_args(argv)))
    config.validate()
    return config
