#!/usr/bin/env python3
"""Mine hard video frames and prepare editable YOLO segmentation annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("scan", "build", "all"), default="all")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--object-confidence", type=float, default=0.25)
    parser.add_argument("--mining-object-confidence", type=float, default=0.15)
    parser.add_argument("--rail-confidence", type=float, default=0.25)
    parser.add_argument("--annotation-confidence", type=float, default=0.15)
    parser.add_argument(
        "--switch-start",
        type=float,
        default=None,
        help="Optional start of a priority scene in seconds",
    )
    parser.add_argument(
        "--switch-end",
        type=float,
        default=None,
        help="Optional end of a priority scene in seconds",
    )
    parser.add_argument("--switch-stride", type=int, default=10)
    parser.add_argument("--max-short-object-frames", type=int, default=120)
    parser.add_argument("--max-rail-hard-frames", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--polygon-epsilon", type=float, default=2.0)
    parser.add_argument("--scan-batch", type=int, default=8)
    parser.add_argument("--selection-mode", choices=("balanced", "all-short"), default="balanced")
    parser.add_argument("--short-track-max-length", type=int, default=3)
    parser.add_argument("--cache-annotations", action="store_true",
                        help="Reuse the exact scan masks when preparing the editor frames")
    parser.add_argument("--dataset-name", default="ReaktRailAi hard-case review")
    return parser.parse_args()


def bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def mask_iou(first: Any, second: Any, np: Any) -> float:
    union = int(np.count_nonzero((first > 0) | (second > 0)))
    if not union:
        return 1.0
    return float(np.count_nonzero((first > 0) & (second > 0)) / union)


def geometry_parts(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    output = []
    for item in getattr(geometry, "geoms", []):
        output.extend(geometry_parts(item))
    return output


def raw_rail_polygons(result: Any, threshold: float, Polygon: Any, make_valid: Any) -> tuple[list[Any], int]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or not hasattr(masks, "xy"):
        return [], 0
    polygons: list[Any] = []
    invalid = 0
    for points, class_id, confidence in zip(masks.xy, boxes.cls, boxes.conf):
        if result.names[int(class_id)] != "rail" or float(confidence) < threshold or len(points) < 4:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            invalid += 1
            polygon = make_valid(polygon)
        polygons.extend(geometry_parts(polygon))
    return polygons, invalid


def polygons_to_mask(polygons: list[Any], height: int, width: int, cv2: Any, np: Any) -> Any:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon.exterior.coords, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def component_count(mask: Any, cv2: Any) -> int:
    count, _labels = cv2.connectedComponents(mask.astype("uint8"))
    return max(0, count - 1)


def extract_objects(result: Any, threshold: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections = []
    for coordinates, class_id, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
        value = float(confidence)
        class_name = result.names[int(class_id)]
        if class_name == "rail" or value < threshold:
            continue
        detections.append(
            {
                "bbox": tuple(float(item) for item in coordinates.tolist()),
                "class_id": int(class_id),
                "class": class_name,
                "confidence": value,
            }
        )
    return detections


class RawRunTracker:
    """Build uninterrupted same-class IoU tracks without persistence."""

    def __init__(self) -> None:
        self.active: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []
        self.next_id = 0

    def update(self, detections: list[dict[str, Any]], frame_index: int) -> None:
        available = set(self.active)
        assignments: list[tuple[dict[str, Any], int]] = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            candidates = [
                (bbox_iou(detection["bbox"], self.active[track_id]["last_bbox"]), track_id)
                for track_id in available
                if self.active[track_id]["class"] == detection["class"]
            ]
            best_iou, best_id = max(candidates, default=(0.0, -1))
            if best_iou > 0.3:
                available.remove(best_id)
                assignments.append((detection, best_id))
            else:
                assignments.append((detection, self.next_id))
                self.next_id += 1

        for track_id in available:
            self.completed.append(self.active.pop(track_id))

        for detection, track_id in assignments:
            if track_id not in self.active:
                self.active[track_id] = {
                    "track_id": track_id,
                    "class_id": detection["class_id"],
                    "class": detection["class"],
                    "start_frame": frame_index,
                    "hits": [],
                }
            track = self.active[track_id]
            track["hits"].append(
                {
                    "frame": frame_index,
                    "confidence": detection["confidence"],
                    "bbox": [round(value, 2) for value in detection["bbox"]],
                }
            )
            track["end_frame"] = frame_index
            track["last_bbox"] = detection["bbox"]

    def break_runs(self) -> None:
        """End all tracks when the source timestamp contains a frame gap."""
        self.completed.extend(self.active.values())
        self.active = {}

    def finish(self) -> list[dict[str, Any]]:
        self.completed.extend(self.active.values())
        self.active = {}
        output = []
        for track in sorted(self.completed, key=lambda item: item["track_id"]):
            track.pop("last_bbox", None)
            confidences = [item["confidence"] for item in track["hits"]]
            output.append(
                {
                    **track,
                    "length": len(track["hits"]),
                    "mean_confidence": statistics.fmean(confidences),
                    "max_confidence": max(confidences),
                }
            )
        return output


def rail_hardness(record: dict[str, Any]) -> float:
    iou = record["selected_iou_previous"]
    return (
        (120.0 if record["raw_rail_parts"] == 0 else 0.0)
        + (100.0 if record["selected_rail_parts"] == 0 else 0.0)
        + (60.0 if record["selected_bottom_pixels"] == 0 else 0.0)
        + max(0, record["selected_components"] - 1) * 8.0
        + record["invalid_raw_rail_parts"] * 2.0
        + (0.0 if iou is None else max(0.0, 0.90 - iou) * 35.0)
    )


def model_and_dependencies(args: argparse.Namespace):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from shapely.geometry import Polygon  # type: ignore
    from shapely.validation import make_valid  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from rail_ai.pipeline import RailPipeline

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/ROCm GPU is required")
    model = YOLO(str(args.model), task="segment")
    selector = RailPipeline.__new__(RailPipeline)
    selector.cv2 = cv2
    selector.np = np
    selector.Polygon = Polygon
    selector.make_valid = make_valid
    selector.config = SimpleNamespace(rail_confidence=args.rail_confidence)
    return cv2, np, torch, Polygon, make_valid, model, selector


def scan_video(args: argparse.Namespace) -> None:
    cv2, np, torch, Polygon, make_valid, model, selector = model_and_dependencies(args)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    records: list[dict[str, Any]] = []
    operating_tracker = RawRunTracker()
    mining_tracker = RawRunTracker()
    previous_mask = None
    previous_timestamp = None
    inference_ms: list[float] = []
    started_scan = time.perf_counter()
    frame_index = 0
    try:
        while frame_index < total:
            frames = []
            timestamps = []
            while len(frames) < max(1, args.scan_batch) and frame_index + len(frames) < total:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
                timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            if not frames:
                break
            torch.cuda.synchronize()
            started = time.perf_counter()
            results = model.predict(
                frames,
                imgsz=args.imgsz,
                device=args.device,
                conf=min(args.mining_object_confidence, args.rail_confidence),
                half=args.precision == "fp16",
                batch=len(frames),
                verbose=False,
            )
            torch.cuda.synchronize()
            per_frame_ms = (time.perf_counter() - started) * 1000.0 / len(frames)
            inference_ms.extend([per_frame_ms] * len(frames))
            for frame, result, source_seconds in zip(frames, results, timestamps):
                if (
                    previous_timestamp is not None
                    and source_seconds - previous_timestamp > max(0.05, 1.5 / fps)
                ):
                    mining_tracker.break_runs()
                    operating_tracker.break_runs()
                mining_detections = extract_objects(result, args.mining_object_confidence)
                operating_detections = [
                    item for item in mining_detections
                    if item["confidence"] >= args.object_confidence
                ]
                mining_tracker.update(mining_detections, frame_index)
                operating_tracker.update(operating_detections, frame_index)
                raw_polygons, invalid_count = raw_rail_polygons(
                    result, args.rail_confidence, Polygon, make_valid
                )
                selected_polygons = selector.extract_rail_polygons(result, frame.shape)
                selected_mask = polygons_to_mask(selected_polygons, height, width, cv2, np)
                selected_iou = None if previous_mask is None else mask_iou(previous_mask, selected_mask, np)
                record = {
                    "frame": frame_index,
                    "seconds": source_seconds,
                    "objects": [
                        {
                            **item,
                            "bbox": [round(value, 2) for value in item["bbox"]],
                            "confidence": round(item["confidence"], 6),
                            "at_operating_threshold": item["confidence"] >= args.object_confidence,
                        }
                        for item in mining_detections
                    ],
                    "raw_rail_parts": len(raw_polygons),
                    "invalid_raw_rail_parts": invalid_count,
                    "selected_rail_parts": len(selected_polygons),
                    "selected_components": component_count(selected_mask, cv2),
                    "selected_bottom_pixels": int(np.count_nonzero(selected_mask[-1])),
                    "selected_rail_pixels": int(np.count_nonzero(selected_mask)),
                    "selected_iou_previous": selected_iou,
                }
                record["rail_hardness"] = rail_hardness(record)
                if args.cache_annotations:
                    record["annotation_instances"] = extract_annotation_instances(
                        result, cv2, np, args.polygon_epsilon
                    )
                records.append(record)
                previous_mask = selected_mask
                previous_timestamp = source_seconds
                frame_index += 1
                if frame_index % 250 == 0:
                    elapsed = time.perf_counter() - started_scan
                    speed = frame_index / max(elapsed, 1e-6)
                    remaining = (total - frame_index) / max(speed, 1e-6)
                    print(
                        f"scan {frame_index}/{total} frames "
                        f"({speed:.1f} fps, ETA {remaining / 60:.1f} min)",
                        flush=True,
                    )
    finally:
        capture.release()

    operating_tracks = operating_tracker.finish()
    mining_tracks = mining_tracker.finish()
    for track in operating_tracks:
        track["threshold"] = args.object_confidence
    for track in mining_tracks:
        track["threshold"] = args.mining_object_confidence
    tracks = sorted(
        operating_tracks + (mining_tracks if args.mining_object_confidence != args.object_confidence else []),
        key=lambda item: (item["threshold"], item["track_id"]),
        reverse=True,
    )
    measured = inference_ms[min(50, len(inference_ms)):]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scan-frames.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "scan-object-tracks.json").write_text(
        json.dumps(tracks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    short = [track for track in tracks if 1 <= track["length"] <= args.short_track_max_length]
    summary = {
        "video": str(args.video),
        "model": str(args.model),
        "model_sha256": model_sha256(args.model),
        "model_imgsz": args.imgsz,
        "precision": args.precision,
        "scan_batch": args.scan_batch,
        "annotation_confidence": min(args.mining_object_confidence, args.rail_confidence),
        "cached_annotations": args.cache_annotations,
        "short_track_max_length": args.short_track_max_length,
        "reported_frames": total,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(records),
        "duration_seconds": (records[-1]["seconds"] + 1.0 / fps) if records else 0.0,
        "object_confidence": args.object_confidence,
        "mining_object_confidence": args.mining_object_confidence,
        "rail_confidence": args.rail_confidence,
        "tracks": len(tracks),
        "short_tracks": len(short),
        "short_tracks_by_threshold": {
            str(threshold): {
                "total": sum(item["threshold"] == threshold for item in short),
                "by_class": dict(sorted(Counter(
                    item["class"] for item in short if item["threshold"] == threshold
                ).items())),
            }
            for threshold in (args.mining_object_confidence, args.object_confidence)
        },
        "inference_ms_mean": statistics.fmean(measured) if measured else 0.0,
        "scan_wall_seconds": time.perf_counter() - started_scan,
    }
    (args.output_dir / "scan-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def add_reason(candidates: dict[int, dict[str, Any]], frame: int, reason: dict[str, Any]) -> None:
    item = candidates.setdefault(frame, {"frame": frame, "reasons": []})
    item["reasons"].append(reason)


def model_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def add_short_track(candidates: dict[int, dict[str, Any]], track: dict[str, Any], threshold: float) -> None:
    threshold = track.get("threshold", threshold)
    for hit in track["hits"]:
        add_reason(candidates, hit["frame"], {
            "type": "short_object",
            "label": f'{track["class"]}: nur {track["length"]} Roh-Frame(s) bei >= {threshold:.2f}',
            "class_id": track["class_id"],
            "class": track["class"],
            "track_id": track["track_id"],
            "run_length": track["length"],
            "threshold": threshold,
            "confidence": hit["confidence"],
            "bbox": hit["bbox"],
        })


def choose_candidates(args: argparse.Namespace, records: list[dict[str, Any]], tracks: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    if args.selection_mode == "all-short":
        for track in tracks:
            if 1 <= track["length"] <= args.short_track_max_length:
                add_short_track(candidates, track, args.object_confidence)
        return materialize_candidates(candidates, records)

    switch_records = []
    if args.switch_start is not None and args.switch_end is not None:
        switch_records = [
            record for record in records
            if args.switch_start <= record["seconds"] <= args.switch_end
        ]
    for record in switch_records[::max(1, args.switch_stride)]:
        frame = record["frame"]
        add_reason(
            candidates,
            frame,
            {"type": "switch", "label": "Weichenszene", "seconds": record["seconds"]},
        )

    short_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        if 1 <= track["length"] <= args.short_track_max_length:
            short_by_class[track["class"]].append(track)
    for items in short_by_class.values():
        items.sort(
            key=lambda item: (
                item.get("threshold", 0.0), item["max_confidence"], args.short_track_max_length + 1 - item["length"]
            ),
            reverse=True,
        )

    queues: dict[str, list[dict[str, Any]]] = {}
    for class_name, items in short_by_class.items():
        selected = []
        centers: list[int] = []
        for item in items:
            center = item["hits"][len(item["hits"]) // 2]["frame"]
            if any(abs(center - other) < 10 for other in centers):
                continue
            selected.append(item)
            centers.append(center)
        queues[class_name] = selected

    short_added = 0
    class_order = sorted(queues, key=lambda name: len(queues[name]), reverse=True)
    positions = {name: 0 for name in class_order}
    while short_added < args.max_short_object_frames:
        progressed = False
        for class_name in class_order:
            position = positions[class_name]
            if position >= len(queues[class_name]):
                continue
            track = queues[class_name][position]
            positions[class_name] += 1
            progressed = True
            hit_frames = [hit["frame"] for hit in track["hits"]]
            new_cost = sum(frame not in candidates for frame in hit_frames)
            if short_added and short_added + new_cost > args.max_short_object_frames:
                continue
            add_short_track(candidates, track, args.object_confidence)
            short_added += new_cost
            if short_added >= args.max_short_object_frames:
                break
        if not progressed:
            break

    hard_records = sorted(records, key=lambda item: item["rail_hardness"], reverse=True)
    hard_selected: list[int] = []
    for record in hard_records:
        if len(hard_selected) >= args.max_rail_hard_frames:
            break
        frame = record["frame"]
        if record["rail_hardness"] <= 0:
            break
        if any(abs(frame - other) < 45 for other in hard_selected):
            continue
        if frame in candidates:
            add_reason(
                candidates,
                frame,
                {
                    "type": "rail_hard",
                    "label": "Auffaellige Gleismaske",
                    "score": record["rail_hardness"],
                },
            )
            hard_selected.append(frame)
            continue
        if any(abs(frame - existing) < 5 for existing in candidates):
            continue
        add_reason(
            candidates,
            frame,
            {
                "type": "rail_hard",
                "label": "Auffaellige Gleismaske",
                "score": record["rail_hardness"],
            },
        )
        hard_selected.append(frame)

    return materialize_candidates(candidates, records)


def materialize_candidates(candidates: dict[int, dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for frame, candidate in sorted(candidates.items()):
        record = records[frame]
        output.append(
            {
                **candidate,
                "seconds": record["seconds"],
                "rail_metrics": {
                    key: record[key]
                    for key in (
                        "raw_rail_parts",
                        "invalid_raw_rail_parts",
                        "selected_rail_parts",
                        "selected_components",
                        "selected_bottom_pixels",
                        "selected_rail_pixels",
                        "selected_iou_previous",
                        "rail_hardness",
                    )
                },
            }
        )
    return output


def class_mapping(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def extract_annotation_instances(result: Any, cv2: Any, np: Any, epsilon: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or not hasattr(masks, "xy"):
        return []
    instances = []
    for index, (points, coordinates, class_id, confidence) in enumerate(
        zip(masks.xy, boxes.xyxy, boxes.cls, boxes.conf)
    ):
        contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(contour) < 3:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon=max(0.0, epsilon), closed=True)
        simplified = simplified.reshape(-1, 2)
        if len(simplified) < 3:
            simplified = contour.reshape(-1, 2)
        instances.append(
            {
                "id": f"model-{index}",
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": round(float(confidence), 6),
                "source": "model",
                "geometry": "polygon",
                "points": [[round(float(x), 1), round(float(y), 1)] for x, y in simplified],
                "bbox": [round(float(value), 1) for value in coordinates.tolist()],
            }
        )
    return instances


def build_dataset(args: argparse.Namespace) -> None:
    scan_summary = json.loads((args.output_dir / "scan-summary.json").read_text(encoding="utf-8"))
    if scan_summary.get("model_sha256", model_sha256(args.model)) != model_sha256(args.model):
        raise ValueError("The selected model differs from the checkpoint used for the scan")
    if args.cache_annotations and not scan_summary.get("cached_annotations"):
        raise ValueError("The scan contains no cached annotations")
    if args.cache_annotations and args.annotation_confidence < scan_summary["annotation_confidence"]:
        raise ValueError("Cached annotations cannot supply detections below the scan confidence")
    if (args.output_dir / "candidates.json").exists():
        raise FileExistsError("Refusing to overwrite an existing review dataset")
    records = read_jsonl(args.output_dir / "scan-frames.jsonl")
    tracks = json.loads((args.output_dir / "scan-object-tracks.json").read_text(encoding="utf-8"))
    fps = float(scan_summary["fps"])
    candidates = choose_candidates(args, records, tracks, fps)
    selected = {item["frame"]: item for item in candidates}
    cv2, np, torch, _Polygon, _make_valid, model, _selector = model_and_dependencies(args)
    images_dir = args.output_dir / "images"
    annotations_dir = args.output_dir / "annotations"
    labels_dir = args.output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    completed = 0
    candidate_entries = []
    try:
        frame_index = 0
        last_selected = max(selected, default=-1)
        while frame_index <= last_selected:
            ok, frame = capture.read()
            if not ok:
                break
            candidate = selected.get(frame_index)
            if candidate is not None:
                if args.cache_annotations:
                    instances = [item for item in records[frame_index]["annotation_instances"]
                                 if item["confidence"] >= args.annotation_confidence]
                else:
                    result = model.predict(
                        frame,
                        imgsz=args.imgsz,
                        device=args.device,
                        conf=args.annotation_confidence,
                        half=args.precision == "fp16",
                        verbose=False,
                    )[0]
                    instances = extract_annotation_instances(result, cv2, np, args.polygon_epsilon)
                candidate_id = f"frame-{frame_index:06d}"
                image_name = f"{candidate_id}.jpg"
                written = cv2.imwrite(
                    str(images_dir / image_name),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, max(70, min(100, args.jpeg_quality))],
                )
                if not written:
                    raise OSError(f"Could not write review image: {image_name}")
                annotation = {
                    "schema_version": 1,
                    "id": candidate_id,
                    "image": f"images/{image_name}",
                    "source_video": str(args.video),
                    "source_frame": frame_index,
                    "source_seconds": candidate["seconds"],
                    "width": width,
                    "height": height,
                    "reviewed": False,
                    "excluded": False,
                    "notes": "",
                    "reasons": candidate["reasons"],
                    "rail_metrics": candidate["rail_metrics"],
                    "instances": instances,
                }
                annotation_path = annotations_dir / f"{candidate_id}.json"
                annotation_path.write_text(
                    json.dumps(annotation, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                entry = {
                    "id": candidate_id,
                    "image": annotation["image"],
                    "annotation": f"annotations/{candidate_id}.json",
                    "source_frame": frame_index,
                    "source_seconds": annotation["source_seconds"],
                    "reviewed": False,
                    "excluded": False,
                    "reasons": candidate["reasons"],
                }
                candidate_entries.append(entry)
                completed += 1
                if completed % 20 == 0 or completed == len(candidates):
                    print(f"build {completed}/{len(candidates)} candidates", flush=True)
            frame_index += 1
    finally:
        capture.release()

    if completed != len(candidates):
        raise RuntimeError(f"Decoded only {completed} of {len(candidates)} selected review frames")
    names = class_mapping(model.names)
    manifest = {
        "schema_version": 1,
        "name": args.dataset_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video": str(args.video),
        "model": str(args.model),
        "model_sha256": model_sha256(args.model),
        "cached_scan_annotations": args.cache_annotations,
        "model_imgsz": args.imgsz,
        "scan_object_confidence": args.object_confidence,
        "mining_object_confidence": args.mining_object_confidence,
        "scan_rail_confidence": args.rail_confidence,
        "annotation_confidence": args.annotation_confidence,
        "width": width,
        "height": height,
        "fps": fps,
        "classes": [{"id": key, "name": value} for key, value in sorted(names.items())],
        "selection": {
            "mode": args.selection_mode,
            "short_track_max_length": args.short_track_max_length,
            "switch_start": args.switch_start,
            "switch_end": args.switch_end,
            "switch_stride": args.switch_stride,
            "max_short_object_frames": args.max_short_object_frames,
            "max_rail_hard_frames": args.max_rail_hard_frames,
        },
        "candidates": candidate_entries,
    }
    (args.output_dir / "candidates.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reason_counts = Counter(
        reason["type"] for item in candidate_entries for reason in item["reasons"]
    )
    build_summary = {
        "candidates": len(candidate_entries),
        "reason_counts": dict(sorted(reason_counts.items())),
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "reviewed_labels_dir": str(labels_dir),
    }
    (args.output_dir / "build-summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(build_summary, indent=2, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.short_track_max_length < 1:
        raise SystemExit("short-track-max-length must be at least 1")
    if args.mining_object_confidence > args.object_confidence:
        raise SystemExit("mining-object-confidence must not exceed object-confidence")
    if (args.switch_start is None) != (args.switch_end is None):
        raise SystemExit("switch-start and switch-end must be provided together")
    if args.switch_start is not None and args.switch_start > args.switch_end:
        raise SystemExit("switch-start must not be greater than switch-end")
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"scan", "all"}:
        scan_video(args)
    if args.stage in {"build", "all"}:
        build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
