"""Per-frame latency metrics."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any


COLUMNS = [
    "frame_id",
    "source_ts_ns",
    "receive_ts_ns",
    "inference_start_ts_ns",
    "inference_end_ts_ns",
    "output_ts_ns",
    "decode_ms",
    "queue_ms",
    "inference_ms",
    "postprocess_ms",
    "publish_enqueue_ms",
    "encode_ms",
    "pipeline_ms",
    "source_to_output_ms",
    "dropped_frames",
    "gpu_utilization_percent",
]


def delta_ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start) / 1_000_000.0)


class LatencyLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: dict[str, Any]) -> None:
        if not self.writer or not self.file:
            return
        with self.lock:
            self.writer.writerow({key: _format(row.get(key)) for key in COLUMNS})
            self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()


def _format(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else value
