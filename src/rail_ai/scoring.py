"""Rail geometry, object tracking, and danger scoring.

The danger score was originally developed for my bachelor's thesis. It is
intended to help a controller decide whether braking is necessary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


CENTER_OFFSET_TOLERANCE = 200


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def compute_danger_score(rail_distance: float, cam_distance: float, movement_delta: float, frame_height: int) -> float:
    base_score = max(0.0, 100.0 - rail_distance)
    y_max = frame_height - cam_distance
    cam_score = (y_max / frame_height) * 100.0
    movement_score = 20.0 if movement_delta > 0 else (-20.0 if movement_delta < 0 else 0.0)
    return max(0.0, min(100.0, (base_score + cam_score + movement_score) * 0.5))


def compute_rail_polygon_score(
    polygon: Any,
    frame_height: int,
    frame_width: int,
    cv2: Any,
    np: Any,
    polygon_type: Any,
) -> tuple[float, float, list[tuple[float, float]]]:
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    frame_polygon = polygon_type([(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)])
    polygon = polygon.intersection(frame_polygon)
    if polygon.is_empty:
        return 100.0, 0.0, []
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)

    min_x, min_y, max_x, max_y = polygon.bounds
    polygon_height = max_y - min_y
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon.exterior.coords, dtype=np.int32)], 255)
    coords = np.array(polygon.exterior.coords)
    bottom = coords[coords[:, 1] >= np.max(coords[:, 1]) - 5]
    x_start = max(0.0, float(np.min(bottom[:, 0]))) if len(bottom) else max(0.0, min_x)
    x_end = min(float(frame_width - 1), float(np.max(bottom[:, 0]))) if len(bottom) else min(float(frame_width - 1), max_x)

    sample_points: list[tuple[float, float]] = []
    extents: list[float] = []
    for x in np.linspace(x_start, x_end, 50):
        x_int = max(0, min(frame_width - 1, int(x)))
        occupied = np.flatnonzero(mask[:, x_int])
        if len(occupied) == 0:
            continue
        start_y = int(occupied[-1])
        column = mask[: start_y + 1, x_int]
        zeros_from_bottom = np.flatnonzero(column[::-1] == 0)
        extent = int(zeros_from_bottom[0]) if len(zeros_from_bottom) else start_y + 1
        sample_points.append((float(x), float(extent)))
        extents.append(float(extent))

    minima = [
        extents[index]
        for index in range(1, len(extents) - 1)
        if extents[index] < extents[index - 1] and extents[index] < extents[index + 1]
    ]
    effective_height = min(minima) if minima else polygon_height
    t1 = (500 / 720) * frame_height
    t2 = (240 / 720) * frame_height
    t3 = (100 / 720) * frame_height
    if effective_height >= t1:
        score = 0.0
    elif effective_height > t2:
        score = ((t1 - effective_height) / (t1 - t2)) * 90.0
    elif effective_height > t3:
        score = 90.0 + ((t2 - effective_height) / (t2 - t3)) * 10.0
    else:
        score = 100.0
    return float(score), float(effective_height), sample_points


@dataclass
class Track:
    bbox: tuple[float, float, float, float]
    last_distance: float | None
    class_name: str
    confidence: float
    score: float
    intersects: bool
    last_seen_ns: int


class DangerTracker:
    def __init__(self, polygon_type: Any, hold_seconds: float = 1.0) -> None:
        self.polygon_type = polygon_type
        self.hold_ns = max(0, round(hold_seconds * 1_000_000_000))
        self.active: dict[int, Track] = {}
        self.next_id = 0

    def update(
        self,
        detections: list[tuple[float, float, float, float, float, str]],
        rail_polygons: list[Any],
        frame_height: int,
        timestamp_ns: int | None = None,
    ) -> tuple[bool, list[list[Any]], dict[int, float], float]:
        current_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        for track_id, track in list(self.active.items()):
            if current_ns - track.last_seen_ns > self.hold_ns:
                self.active.pop(track_id, None)

        unmatched = set(self.active)
        matched: list[tuple[tuple[float, float, float, float, float, str], int]] = []
        for detection in detections:
            bbox = detection[:4]
            best_id = None
            best_iou = 0.0
            for track_id in unmatched:
                if self.active[track_id].class_name != detection[5]:
                    continue
                value = compute_iou(self.active[track_id].bbox, bbox)
                if value > best_iou:
                    best_iou = value
                    best_id = track_id
            if best_id is not None and best_iou > 0.3:
                unmatched.remove(best_id)
                matched.append((detection, best_id))
            else:
                matched.append((detection, self.next_id))
                self.next_id += 1

        any_intersection = False
        boxes: list[list[Any]] = []
        danger_scores: dict[int, float] = {}
        for detection, track_id in matched:
            x_min, y_min, x_max, y_max, confidence, class_name = detection
            previous = self.active.get(track_id)
            box_polygon = self.polygon_type([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
            rail_distance = min((box_polygon.distance(rail) for rail in rail_polygons), default=9999.0)
            movement = (previous.last_distance - rail_distance) if previous and previous.last_distance is not None else 0.0
            score = compute_danger_score(rail_distance, frame_height - y_max, movement, frame_height)
            intersects = bool(rail_polygons and any(box_polygon.intersects(rail) for rail in rail_polygons))
            if intersects:
                score = max(score, 80.0)
                if y_max > (2 / 3) * frame_height and abs(movement) < 1e-2:
                    score = 100.0
            any_intersection = any_intersection or intersects
            danger_scores[track_id] = score
            boxes.append([x_min, y_min, x_max, y_max, confidence, class_name, score, intersects])
            self.active[track_id] = Track(
                bbox=(x_min, y_min, x_max, y_max),
                last_distance=rail_distance,
                class_name=class_name,
                confidence=confidence,
                score=score,
                intersects=intersects,
                last_seen_ns=current_ns,
            )

        # Missing detections remain visible for a short, time-based hold period.
        # They intentionally do not contribute to the current safety score:
        # persistence smooths the overlay without inventing a fresh detection.
        current_bboxes = [detection[:4] for detection, _track_id in matched]
        held_bboxes: list[tuple[float, float, float, float]] = []
        # Prefer the most recently seen, most confident held box when two stale
        # tracks overlap. Any held box overlapping a current detection is a
        # visual ghost and disappears immediately instead of waiting one second.
        held_ids = sorted(
            unmatched,
            key=lambda track_id: (
                self.active[track_id].last_seen_ns,
                self.active[track_id].confidence,
                track_id,
            ),
            reverse=True,
        )
        for track_id in held_ids:
            track = self.active[track_id]
            if any(compute_iou(track.bbox, bbox) > 0.0 for bbox in current_bboxes + held_bboxes):
                self.active.pop(track_id, None)
                continue
            boxes.append(
                [
                    *track.bbox,
                    track.confidence,
                    track.class_name,
                    track.score,
                    track.intersects,
                ]
            )
            held_bboxes.append(track.bbox)

        overall = max(danger_scores.values(), default=0.0)
        return any_intersection, boxes, danger_scores, overall
