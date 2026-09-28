"""Annotated WebRTC transport output and optional recording."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class OutputError(RuntimeError):
    """The annotated output could not be published."""


class HardwareCodecError(OutputError):
    """An explicitly requested hardware codec could not be used."""


def command_supports(name: str, arguments: list[str], needle: str) -> bool:
    if not shutil.which(name):
        return False
    result = subprocess.run([name, *arguments], capture_output=True, text=True, check=False)
    return needle in result.stdout or needle in result.stderr


class WebRTCWriter:
    """Encode annotated frames to H.264/RTP for the in-container WebRTC server."""

    def __init__(self, config: Any, np: Any, vaapi_device: str = "") -> None:
        self.config = config
        self.np = np
        self.process: subprocess.Popen[bytes] | None = None
        self.backend = "disabled"
        self._software_retried = False
        self._condition = threading.Condition()
        self._latest_frame: Any | None = None
        self._stop = False
        self._worker: threading.Thread | None = None
        self._error: Exception | None = None
        self._process_started_at: float | None = None
        self._rapid_software_failures = 0
        self.dropped_frames = 0
        self.reconnects = 0
        self.last_encode_ms: float | None = None
        self.fallback_reason = ""
        self.vaapi_device = vaapi_device

    def _encoder_candidates(self) -> list[str]:
        if self.config.webrtc_encoder != "auto":
            return [self.config.webrtc_encoder]
        candidates = []
        if self.config.platform == "jetson":
            candidates.append("nvv4l2")
        if self.config.platform == "rtx":
            candidates.append("nvenc")
        if self.config.platform == "amd":
            candidates.append("vaapi")
        candidates.append("software")
        return candidates

    def _choose_encoder(self) -> str:
        unavailable: list[str] = []
        for candidate in self._encoder_candidates():
            if candidate == "nvv4l2" and command_supports("gst-inspect-1.0", ["nvv4l2h264enc"], "nvv4l2h264enc"):
                return candidate
            if candidate == "nvenc" and command_supports("ffmpeg", ["-encoders"], "h264_nvenc"):
                return candidate
            if (
                candidate == "vaapi"
                and self.vaapi_device
                and Path(self.vaapi_device).exists()
                and command_supports("ffmpeg", ["-encoders"], "h264_vaapi")
            ):
                return candidate
            if candidate == "software" and shutil.which("ffmpeg"):
                if unavailable:
                    self.fallback_reason = ", ".join(unavailable)
                return candidate
            if candidate != "software":
                unavailable.append(f"{candidate} unavailable")
        message = "No supported H.264 encoder is installed"
        if self.config.webrtc_encoder not in {"auto", "software"}:
            raise HardwareCodecError(
                f"Explicit hardware encoder {self.config.webrtc_encoder} is unavailable"
            )
        raise OutputError(message)

    def _ffmpeg_command(self, width: int, height: int, encoder: str) -> list[str]:
        codec = "h264_nvenc" if encoder == "nvenc" else "h264_vaapi" if encoder == "vaapi" else "libx264"
        if encoder == "nvenc":
            codec_args = [
                "-preset",
                "p4",
                "-tune",
                "ull",
                "-rc",
                "cbr",
                "-bf",
                "0",
                "-profile:v",
                "baseline",
                "-level:v",
                "4.1",
            ]
        elif encoder == "vaapi":
            codec_args = [
                "-vf",
                "format=nv12,hwupload",
                "-quality",
                "0",
                "-rc_mode",
                "CBR",
                "-bf",
                "0",
                "-profile:v",
                "constrained_baseline",
                "-level:v",
                "4.1",
            ]
        else:
            codec_args = [
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-bf",
                "0",
                "-profile:v",
                "baseline",
                "-level:v",
                "4.1",
                "-x264-params",
                "repeat-headers=1:scenecut=0:nal-hrd=cbr",
            ]
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
        ]
        if encoder == "vaapi":
            command.extend(["-vaapi_device", self.vaapi_device])
        output_pixel_format = [] if encoder == "vaapi" else ["-pix_fmt", "yuv420p"]
        command.extend([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.config.output_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
            *codec_args,
            "-g",
            str(max(1, round(self.config.output_fps))),
            "-b:v",
            f"{self.config.h264_bitrate_kbps}k",
            "-minrate",
            f"{self.config.h264_bitrate_kbps}k",
            "-maxrate",
            f"{self.config.h264_bitrate_kbps}k",
            "-bufsize",
            f"{self.config.h264_bitrate_kbps}k",
            "-bsf:v",
            "dump_extra=freq=keyframe",
            *output_pixel_format,
            "-payload_type",
            "96",
            "-f",
            "rtp",
            self.config.webrtc_rtp_url,
        ])
        return command

    def _gstreamer_command(self, width: int, height: int) -> list[str]:
        host_port = self.config.webrtc_rtp_url.split("://", 1)[1].split("?", 1)[0]
        host, port = host_port.rsplit(":", 1)
        return [
            "gst-launch-1.0",
            "-q",
            "fdsrc",
            "fd=0",
            "do-timestamp=true",
            f"blocksize={width * height * 3}",
            "!",
            "rawvideoparse",
            "format=bgr",
            f"width={width}",
            f"height={height}",
            f"framerate={round(self.config.output_fps)}/1",
            "!",
            "queue",
            "leaky=downstream",
            "max-size-buffers=1",
            "!",
            "videoconvert",
            "!",
            "nvvidconv",
            "compute-hw=GPU",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            f"bitrate={self.config.h264_bitrate_kbps * 1000}",
            "control-rate=1",
            "insert-sps-pps=true",
            f"iframeinterval={max(1, round(self.config.output_fps))}",
            f"idrinterval={max(1, round(self.config.output_fps))}",
            "num-B-Frames=0",
            "poc-type=2",
            "copy-timestamp=true",
            "profile=0",
            "!",
            "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,profile=(string)baseline,level=(string)4.1",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "rtph264pay",
            "pt=96",
            "config-interval=1",
            "mtu=1200",
            "!",
            "udpsink",
            f"host={host}",
            f"port={port}",
            "sync=false",
            "async=false",
        ]

    def _start(self, frame: Any, forced_encoder: str | None = None) -> None:
        height, width = frame.shape[:2]
        encoder = forced_encoder or self._choose_encoder()
        command = self._gstreamer_command(width, height) if encoder == "nvv4l2" else self._ffmpeg_command(width, height, encoder)
        self.backend = encoder
        print(f"WebRTC publisher: encoder={encoder} rtp={self.config.webrtc_rtp_url}", flush=True)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self._process_started_at = time.monotonic()

    def _write_frame(self, frame: Any) -> None:
        if self.process is None:
            self._start(frame)
        assert self.process is not None
        if self.process.poll() is not None:
            returncode = self.process.returncode
            self._recover_encoder(frame, f"process exited with status {returncode}")
            if self._stop:
                return
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(self.np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._recover_encoder(frame, f"pipe failed: {exc}")
            if self._stop:
                return
            assert self.process and self.process.stdin
            self.process.stdin.write(self.np.ascontiguousarray(frame).tobytes())

    def _recover_encoder(self, frame: Any, reason: str) -> None:
        failed_backend = self.backend
        process_uptime = (
            0.0
            if self._process_started_at is None
            else time.monotonic() - self._process_started_at
        )
        if (
            self.config.webrtc_encoder == "auto"
            and failed_backend != "software"
            and not self._software_retried
        ):
            self._software_retried = True
            self._rapid_software_failures = 0
            self.fallback_reason = f"{failed_backend} encoder {reason}; using software"
            print(
                f"Hardware encoder {failed_backend} failed ({reason}); retrying with software",
                flush=True,
            )
            self._close_process()
            self._start(frame, "software")
            return
        if failed_backend != "software":
            raise HardwareCodecError(f"Explicit hardware encoder {failed_backend} failed: {reason}")
        if process_uptime < 2.0:
            self._rapid_software_failures += 1
        else:
            self._rapid_software_failures = 0
        if self._rapid_software_failures >= 5:
            raise OutputError(
                f"Software encoder failed {self._rapid_software_failures} times during startup: {reason}"
            )
        self.fallback_reason = f"software encoder {reason}; reconnecting"
        self._reconnect(frame)

    def _reconnect(self, frame: Any) -> None:
        if self._stop:
            return
        self.reconnects += 1
        if self.reconnects == 1 or self.reconnects % 10 == 0:
            print(f"WebRTC publisher reconnect attempt {self.reconnects}", flush=True)
        self._close_process()
        time.sleep(0.2)
        self._start(frame, "software" if self.backend == "software" else None)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._latest_frame is None and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        return
                    frame = self._latest_frame
                    self._latest_frame = None
                started = time.perf_counter_ns()
                self._write_frame(frame)
                self.last_encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        except Exception as exc:
            self._error = exc

    def write(self, frame: Any) -> None:
        if not self.config.webrtc_enabled:
            return
        if self._error is not None:
            if isinstance(self._error, OutputError):
                raise self._error
            raise OutputError(f"WebRTC publisher failed: {self._error}") from self._error
        with self._condition:
            if self._latest_frame is not None:
                self.dropped_frames += 1
            self._latest_frame = frame
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="webrtc-publisher", daemon=True)
                self._worker.start()
            self._condition.notify()

    @property
    def ready(self) -> bool:
        process = self.process
        return (
            self.config.webrtc_enabled
            and process is not None
            and process.poll() is None
            and self._error is None
        )

    def _close_process(self) -> None:
        process, self.process = self.process, None
        self._process_started_at = None
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._latest_frame = None
            self._condition.notify_all()
        self._close_process()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=5)
        self._close_process()


class RecordingWriter:
    def __init__(self, base: Path | None, annotated: bool, fps: float, cv2: Any) -> None:
        self.enabled = base is not None
        self.cv2 = cv2
        self.raw = None
        self.annotated = None
        self.metadata = None
        self.raw_path = None
        self.annotated_path = None
        self.fps = fps
        self.record_annotated = annotated
        if base:
            run_dir = base / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            self.raw_path = run_dir / "raw.mp4"
            self.annotated_path = run_dir / "annotated.mp4"
            self.metadata = (run_dir / "annotations.jsonl").open("w", encoding="utf-8")

    def _open(self, path: Path, frame: Any):
        height, width = frame.shape[:2]
        writer = self.cv2.VideoWriter(
            str(path),
            self.cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open recording file: {path}")
        return writer

    def write(self, raw_frame: Any, annotated_frame: Any, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.raw is None:
            assert self.raw_path is not None
            self.raw = self._open(self.raw_path, raw_frame)
        self.raw.write(raw_frame)
        if self.record_annotated:
            if self.annotated is None:
                assert self.annotated_path is not None
                self.annotated = self._open(self.annotated_path, annotated_frame)
            self.annotated.write(annotated_frame)
        assert self.metadata is not None
        self.metadata.write(json.dumps(metadata, default=str, sort_keys=True) + "\n")
        self.metadata.flush()

    def close(self) -> None:
        for writer in (self.raw, self.annotated):
            if writer is not None:
                writer.release()
        if self.metadata and not self.metadata.closed:
            self.metadata.close()
