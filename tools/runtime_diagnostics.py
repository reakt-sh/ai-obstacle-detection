#!/usr/bin/env python3
"""Report accelerator, ONNX provider and codec availability."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def command_succeeds(command: list[str], timeout: int = 30) -> bool:
    if not shutil.which(command[0]):
        return False
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("cpu", "amd", "rtx", "jetson"), required=True)
    args = parser.parse_args()
    result: dict[str, object] = {"platform": args.platform}

    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_gpu_available"] = bool(torch.cuda.is_available())
        result["torch_hip"] = getattr(torch.version, "hip", None)
        result["torch_cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            tensor = torch.ones((64, 64), device="cuda")
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_warmup"] = float((tensor @ tensor).sum().cpu())
    except Exception as exc:
        result["torch_error"] = str(exc)

    try:
        import onnxruntime as ort

        result["onnxruntime_version"] = ort.__version__
        result["onnx_providers"] = ort.get_available_providers()
    except Exception as exc:
        result["onnxruntime_error"] = str(exc)

    result["ffmpeg"] = bool(shutil.which("ffmpeg"))
    result["nvenc"] = command_succeeds(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
    )
    vaapi_device = os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")
    result["vaapi_device"] = vaapi_device
    result["vaapi"] = command_succeeds(
        [
            "ffmpeg",
            "-v",
            "error",
            "-vaapi_device",
            vaapi_device,
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=1",
            "-vf",
            "format=nv12,hwupload",
            "-frames:v",
            "1",
            "-c:v",
            "h264_vaapi",
            "-f",
            "null",
            "-",
        ]
    )
    result["nvv4l2"] = command_succeeds(
        [
            "gst-launch-1.0",
            "-q",
            "videotestsrc",
            "num-buffers=1",
            "!",
            "video/x-raw,width=64,height=64,format=I420",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            "control-rate=1",
            "insert-sps-pps=true",
            "iframeinterval=1",
            "idrinterval=1",
            "num-B-Frames=0",
            "poc-type=2",
            "copy-timestamp=true",
            "profile=0",
            "!",
            "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,profile=(string)baseline,level=(string)4.1",
            "!",
            "h264parse",
            "!",
            "fakesink",
        ]
    )
    result["tensorrt"] = bool(shutil.which("trtexec"))
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.platform != "cpu" and not result.get("torch_gpu_available"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
