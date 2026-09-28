"""Low-latency RTSP frame reader."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class FramePacket:
    frame_id: int
    frame: Any
    source_ts_ns: int | None
    receive_ts_ns: int
    decode_ms: float | None = None


class LatestFrameBuffer:
    """Single-slot handoff that never lets stale frames queue up."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._last_consumed_id: int | None = None
        self._stopped = False

    def put(self, packet: FramePacket) -> None:
        with self._condition:
            if self._last_consumed_id is not None and packet.frame_id <= self._last_consumed_id:
                return
            if self._latest is not None and packet.frame_id <= self._latest.frame_id:
                return
            self._latest = packet
            self._condition.notify_all()

    def get_latest(self, timeout_s: float = 0.2) -> tuple[FramePacket, int] | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._stopped:
                if self._latest is not None and self._latest.frame_id != self._last_consumed_id:
                    packet = self._latest
                    dropped = 0 if self._last_consumed_id is None else max(0, packet.frame_id - self._last_consumed_id - 1)
                    self._last_consumed_id = packet.frame_id
                    return packet, dropped
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    @property
    def stopped(self) -> bool:
        with self._condition:
            return self._stopped


class FFmpegReader(threading.Thread):
    def __init__(
        self,
        config: Any,
        buffer: LatestFrameBuffer,
        cv2: Any,
        np: Any,
        vaapi_device: str = "",
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.buffer = buffer
        self.cv2 = cv2
        self.np = np
        self.error = ""
        self.decode_backend = "starting"
        self.fallback_reason = ""
        self.vaapi_device = vaapi_device
        self.fps = config.output_fps
        self._next_frame_id = 1

    def _probe(self) -> tuple[int, int, float]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            self.config.input_url,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffprobe failed")
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / max(1.0, float(denominator))
        return int(stream["width"]), int(stream["height"]), fps or self.config.output_fps

    def _software_command(self, _width: int, _height: int) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-avioflags",
            "direct",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-use_wallclock_as_timestamps",
            "1",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.config.input_url,
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]

    def _cuda_command(self, width: int, height: int) -> list[str]:
        command = self._software_command(width, height)
        input_index = command.index("-i")
        command[input_index:input_index] = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        pixel_index = command.index("-pix_fmt")
        command[pixel_index:pixel_index] = ["-vf", "hwdownload,format=nv12,format=bgr24"]
        return command

    def _vaapi_command(self, width: int, height: int) -> list[str]:
        command = self._software_command(width, height)
        input_index = command.index("-i")
        command[input_index:input_index] = [
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            self.vaapi_device,
            "-hwaccel_output_format",
            "vaapi",
        ]
        pixel_index = command.index("-pix_fmt")
        command[pixel_index:pixel_index] = ["-vf", "hwdownload,format=nv12,format=bgr24"]
        return command

    def _nvv4l2_command(self) -> list[str]:
        return [
            "gst-launch-1.0",
            "-q",
            "rtspsrc",
            f"location={self.config.input_url}",
            "latency=0",
            "protocols=tcp",
            "drop-on-latency=true",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            "nvv4l2decoder",
            "!",
            "queue",
            "leaky=downstream",
            "max-size-buffers=1",
            "max-size-bytes=0",
            "max-size-time=0",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw,format=BGRx",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=BGR",
            "!",
            "queue",
            "leaky=downstream",
            "max-size-buffers=1",
            "max-size-bytes=0",
            "max-size-time=0",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]

    def _run_command(self, command: list[str], width: int, height: int, backend: str) -> bool:
        frame_bytes = width * height * 3
        # Let decoder diagnostics reach the container log. Suppressing stderr
        # made Jetson/GStreamer negotiation failures look like a black stream
        # with no actionable error message.
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        self.decode_backend = backend
        read_count = 0
        try:
            assert process.stdout is not None
            while not self.buffer.stopped:
                payload = process.stdout.read(frame_bytes)
                if len(payload) != frame_bytes:
                    break
                frame_id = self._next_frame_id
                self._next_frame_id += 1
                read_count += 1
                # A previous probe failure may still be visible while this
                # long-running decoder process is healthy. Clear it as soon as
                # the reconnected stream yields its first complete frame.
                self.error = ""
                received = now_ns()
                frame = self.np.frombuffer(payload, dtype=self.np.uint8).reshape((height, width, 3)).copy()
                self.buffer.put(FramePacket(frame_id, frame, None, received, None))
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        return read_count > 0

    def run(self) -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.error = "ffmpeg and ffprobe are required for RTSP input"
            self.buffer.stop()
            return
        try:
            while not self.buffer.stopped:
                try:
                    width, height, self.fps = self._probe()
                    decoder = self.config.input_decoder
                    use_nvv4l2 = decoder == "nvv4l2" or (decoder == "auto" and self.config.platform == "jetson")
                    if use_nvv4l2 and shutil.which("gst-launch-1.0"):
                        if self._run_command(self._nvv4l2_command(), width, height, "gstreamer-nvv4l2"):
                            self.error = ""
                            continue
                        if decoder != "auto":
                            self.error = "Explicit nvv4l2 decoder failed"
                            self.buffer.stop()
                            return
                        self.fallback_reason = "nvv4l2 decode failed"
                    elif decoder == "nvv4l2":
                        self.error = "Explicit nvv4l2 decoder requested but GStreamer is unavailable"
                        self.buffer.stop()
                        return
                    elif decoder == "auto" and self.config.platform == "jetson":
                        self.fallback_reason = "GStreamer nvv4l2 decoder is unavailable"

                    use_cuda = decoder == "cuda" or (decoder == "auto" and self.config.platform == "rtx")
                    if use_cuda and self._run_command(self._cuda_command(width, height), width, height, "ffmpeg-cuda"):
                        self.error = ""
                        continue
                    if decoder == "cuda":
                        self.error = "Explicit CUDA decoder failed"
                        self.buffer.stop()
                        return
                    if decoder == "auto" and self.config.platform == "rtx":
                        self.fallback_reason = "CUDA decode failed"

                    use_vaapi = decoder == "vaapi" or (decoder == "auto" and self.config.platform == "amd")
                    if use_vaapi:
                        if not self.vaapi_device:
                            if decoder != "auto":
                                self.error = "Explicit VAAPI decoder requested but no matching render node was found"
                                self.buffer.stop()
                                return
                            self.fallback_reason = "no VAAPI render node matched the ROCm GPU"
                        elif self._run_command(
                            self._vaapi_command(width, height),
                            width,
                            height,
                            "ffmpeg-vaapi",
                        ):
                            self.error = ""
                            continue
                        elif decoder != "auto":
                            self.error = f"Explicit VAAPI decoder failed on {self.vaapi_device}"
                            self.buffer.stop()
                            return
                        else:
                            self.fallback_reason = f"VAAPI decode failed on {self.vaapi_device}"

                    if self._run_command(self._software_command(width, height), width, height, "ffmpeg-software"):
                        self.error = ""
                        continue
                except Exception as exc:
                    self.error = f"RTSP reader failed: {exc}"
                if not self.buffer.stopped:
                    time.sleep(1.0)
        finally:
            self.buffer.stop()


def create_reader(
    config: Any,
    buffer: LatestFrameBuffer,
    cv2: Any,
    np: Any,
    vaapi_device: str = "",
) -> FFmpegReader:
    return FFmpegReader(config, buffer, cv2, np, vaapi_device)
