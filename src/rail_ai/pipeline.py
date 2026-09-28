"""End-to-end rail inference pipeline."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from typing import Any

from .config import AppConfig
from .frames import FramePacket, LatestFrameBuffer, create_reader, now_ns
from .metrics import LatencyLogger, delta_ms
from .models import UnifiedModel, load_model
from .output import RecordingWriter, WebRTCWriter
from .runtime import RuntimeManager
from .scoring import CENTER_OFFSET_TOLERANCE, DangerTracker, compute_rail_polygon_score


RAIL_COLOR = (0, 255, 0)
RAIL_FILL_ALPHA = 0.22
RAIL_ANOMALY_COLOR = (0, 165, 255)
RAIL_ANOMALY_FILL_ALPHA = 0.40
RAIL_MODEL_MASK_HOLD_FRAMES = 2
RAIL_GRAPH_GAP_AT_720P = 12.0
RAIL_BOTTOM_BAND_RATIO = 0.08
ON_RAIL_COLOR = (0, 0, 255)
OFF_RAIL_COLOR = (255, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)


class ModelInferenceError(RuntimeError):
    """A model invocation failed and may justify a runtime fallback."""


class RailPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cv2, self.np, self.Polygon, self.make_valid = self._load_dependencies()
        self.runtime = RuntimeManager(config)
        self.runtime.select()
        self.vaapi_device = self.runtime.state.vaapi_device if config.vaapi_device == "auto" else config.vaapi_device
        self.model = self._load_model_with_fallback()
        self.buffer = LatestFrameBuffer()
        self.reader = create_reader(config, self.buffer, self.cv2, self.np, self.vaapi_device)
        self.tracker = DangerTracker(self.Polygon, config.object_hold_seconds)
        self.cached_polygons: list[Any] = []
        self.rail_mask_stale_frames = 0
        self.processed_frames = 0
        self.total_dropped_frames = 0
        self.last_output_ns: int | None = None
        self.next_output_ns: int | None = None
        self.last_frame_log_ns = 0
        self.gpu_utilization_percent: float | None = None
        self.stop_event = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self.results_lock = threading.Lock()
        self.stream_writer = WebRTCWriter(config, self.np, self.vaapi_device)
        self.recording = RecordingWriter(config.record_dir, config.record_annotated, config.output_fps, self.cv2)
        self.latency_logger = LatencyLogger(config.latency_csv)
        self.latest_frame = None
        self.status: dict[str, Any] = {
            "live": True,
            "ready": False,
            "frame_id": 0,
            "score": 0.0,
            "rail_score": 0.0,
            "rail_anomaly": False,
            "rail_anomaly_y": None,
            "rail_mask_stale_frames": 0,
            "fps": 0.0,
            "dropped_frames": 0,
            "requested_format": config.model_format,
            "effective_format": self.model.format,
            "requested_precision": config.model_precision,
            "effective_precision": self.model.effective_precision,
            "input_protocol": config.input_protocol,
            "input_decoder": "starting",
            "requested_input_decoder": config.input_decoder,
            "decoder_fallback_reason": "",
            "webrtc_enabled": config.webrtc_enabled,
            "webrtc_ready": False,
            "webrtc_encoder": "starting" if config.webrtc_enabled else "disabled",
            "requested_webrtc_encoder": config.webrtc_encoder,
            "encoder_fallback_reason": "",
            "vaapi_device": self.vaapi_device,
            "whep_path": "/processed/whep",
            **self.runtime.state.to_dict(),
        }

    @staticmethod
    def _load_dependencies():
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            from shapely.geometry import Polygon  # type: ignore
            from shapely.validation import make_valid  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Missing runtime dependency; install requirements/runtime.txt") from exc
        return cv2, np, Polygon, make_valid

    def _load_model_with_fallback(self) -> UnifiedModel:
        try:
            return load_model(self.config, self.runtime)
        except Exception as exc:
            if not self.runtime.can_runtime_fallback() or self.config.model_format == "tensorrt":
                raise
            print(f"GPU model initialization failed; falling back to CPU: {exc}", flush=True)
            self.runtime.force_cpu(f"model initialization failed: {exc}")
            return load_model(self.config, self.runtime)

    def fallback_to_cpu(self, reason: str) -> None:
        if not self.runtime.can_runtime_fallback() or self.config.model_format == "tensorrt":
            raise RuntimeError(reason)
        print(f"GPU runtime failed; reloading model on CPU: {reason}", flush=True)
        self.runtime.force_cpu(reason)
        self.model = load_model(self.config, self.runtime)
        self.tracker = DangerTracker(self.Polygon, self.config.object_hold_seconds)
        self.cached_polygons = []
        self.rail_mask_stale_frames = 0
        with self.results_lock:
            self.status.update(self.runtime.state.to_dict())
            self.status["effective_precision"] = self.model.effective_precision

    def start(self) -> None:
        self.reader.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.buffer.stop()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.stop()
        if self.reader.is_alive() and self.reader is not threading.current_thread():
            self.reader.join(timeout=5)
        self.stream_writer.close()
        self.recording.close()
        self.latency_logger.close()
        with self.results_lock:
            self.status["live"] = False
            self.status["ready"] = False
            self.status["webrtc_ready"] = False
        self._write_status_report()

    def _write_status_report(self) -> None:
        if self.config.status_json is None:
            return
        try:
            self.config.status_json.parent.mkdir(parents=True, exist_ok=True)
            self.config.status_json.write_text(
                json.dumps(self.status_snapshot(), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Unable to write status report {self.config.status_json}: {exc}", flush=True)

    def extract_rail_polygons(self, result: Any, frame_shape: tuple[int, int, int]) -> list[Any]:
        height, width = frame_shape[:2]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        polygons: list[Any] = []
        if masks is not None and hasattr(masks, "xy") and boxes is not None:
            for points, class_id, confidence in zip(masks.xy, boxes.cls, boxes.conf):
                if (
                    len(points) < 4
                    or result.names[int(class_id)] != "rail"
                    or float(confidence) < self.config.rail_confidence
                ):
                    continue
                polygon = self.Polygon(points)
                if not polygon.is_valid:
                    polygon = self.make_valid(polygon)
                polygons.extend(self._polygon_parts(polygon))
        if not polygons:
            return []

        minimum_area = max(16.0, width * height * 1e-5)
        polygons = [polygon for polygon in polygons if polygon.area >= minimum_area]
        if not polygons:
            return []

        scale = min(width / 1280.0, height / 720.0)
        graph_gap = max(4.0, RAIL_GRAPH_GAP_AT_720P * scale)
        center_x = width / 2.0
        center_half_width = max(20.0, width * (CENTER_OFFSET_TOLERANCE / 1280.0))
        bottom_band_height = max(12.0, height * RAIL_BOTTOM_BAND_RATIO)
        bottom_center = self.Polygon(
            [
                (center_x - center_half_width, height - bottom_band_height),
                (center_x + center_half_width, height - bottom_band_height),
                (center_x + center_half_width, height - 1),
                (center_x - center_half_width, height - 1),
            ]
        )
        connected = {
            index
            for index, polygon in enumerate(polygons)
            if polygon.distance(bottom_center) <= graph_gap
        }
        frontier = list(connected)
        while frontier:
            current = frontier.pop()
            for index, polygon in enumerate(polygons):
                if index in connected:
                    continue
                if polygons[current].distance(polygon) <= graph_gap:
                    connected.add(index)
                    frontier.append(index)
        selected = [polygon for index, polygon in enumerate(polygons) if index in connected]
        sealed: list[Any] = []
        for polygon in selected:
            sealed.extend(self._seal_rail_polygon_to_bottom(polygon, width, height, graph_gap))
        return self._close_selected_rail_graph(sealed, graph_gap)

    def _close_selected_rail_graph(self, polygons: list[Any], graph_gap: float) -> list[Any]:
        """Join small cracks inside the selected ego-track graph only.

        Model contours can become self-intersecting around points and are then
        split into several valid polygons.  Expanding by half the graph gap,
        taking the union and shrinking again reconnects pieces separated by at
        most ``graph_gap`` pixels.  This runs after graph selection, so a
        nearby parallel track cannot be pulled into the result.
        """
        if len(polygons) < 2:
            return polygons
        radius = max(1.0, graph_gap / 2.0)
        expanded = polygons[0].buffer(radius, quad_segs=2)
        for polygon in polygons[1:]:
            expanded = expanded.union(polygon.buffer(radius, quad_segs=2))
        closed = expanded.buffer(-radius, quad_segs=2)
        if not closed.is_valid:
            closed = self.make_valid(closed)
        return self._polygon_parts(closed)

    def _seal_rail_polygon_to_bottom(
        self,
        polygon: Any,
        frame_width: int,
        frame_height: int,
        graph_gap: float,
    ) -> list[Any]:
        """Extend a near-bottom ego-rail contour cleanly to the image edge."""
        _min_x, _min_y, _max_x, max_y = polygon.bounds
        bottom_y = float(frame_height - 1)
        snap_distance = max(12.0, frame_height * RAIL_BOTTOM_BAND_RATIO)
        if bottom_y - max_y > snap_distance:
            return [polygon]

        coordinates = self.np.asarray(polygon.exterior.coords, dtype=float)
        lower_edge = coordinates[coordinates[:, 1] >= max_y - max(2.0, graph_gap)]
        if len(lower_edge) < 2:
            return [polygon]
        left = max(0.0, float(lower_edge[:, 0].min()))
        right = min(float(frame_width - 1), float(lower_edge[:, 0].max()))
        if right - left < 2.0:
            return [polygon]

        anchor_y = max(0.0, min(bottom_y, max_y - max(2.0, graph_gap)))
        extension = self.Polygon(
            [
                (left, anchor_y),
                (right, anchor_y),
                (right, bottom_y),
                (left, bottom_y),
            ]
        )
        sealed = polygon.union(extension)
        if not sealed.is_valid:
            sealed = self.make_valid(sealed)
        return self._polygon_parts(sealed)

    @staticmethod
    def _polygon_parts(geometry: Any) -> list[Any]:
        if geometry.is_empty:
            return []
        if geometry.geom_type == "Polygon":
            return [geometry]
        parts: list[Any] = []
        for item in getattr(geometry, "geoms", []):
            parts.extend(RailPipeline._polygon_parts(item))
        return parts

    def extract_objects(self, result: Any) -> list[tuple[float, float, float, float, float, str]]:
        detected = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detected
        for coordinates, class_id, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_name = result.names[int(class_id)]
            if class_name == "rail" or float(confidence) < self.config.object_confidence:
                continue
            x_min, y_min, x_max, y_max = coordinates.tolist()
            detected.append((x_min, y_min, x_max, y_max, float(confidence), class_name))
        return detected

    @staticmethod
    def extract_rail_confidence(result: Any) -> float | None:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return None
        confidences = [
            float(confidence)
            for class_id, confidence in zip(boxes.cls, boxes.conf)
            if result.names[int(class_id)] == "rail"
        ]
        return max(confidences, default=None)

    def update_rail_polygons(self, detected_polygons: list[Any]) -> list[Any]:
        """Keep the connected ego-track graph across two transient misses."""
        if detected_polygons:
            self.cached_polygons = detected_polygons
            self.rail_mask_stale_frames = 0
        elif self.cached_polygons and self.rail_mask_stale_frames < RAIL_MODEL_MASK_HOLD_FRAMES:
            self.rail_mask_stale_frames += 1
        else:
            self.cached_polygons = []
            self.rail_mask_stale_frames = 0
        return self.cached_polygons

    def draw(
        self,
        frame: Any,
        polygons: list[Any],
        boxes: list[list[Any]],
        sample_points: list[tuple[float, float]],
        anomaly_y: float | None = None,
    ) -> Any:
        output = frame.copy()
        polygon_points = [
            self.np.array(polygon.exterior.coords, dtype=self.np.int32)
            for polygon in polygons
        ]
        output = self._blend_polygon_fill(output, polygon_points, RAIL_COLOR, RAIL_FILL_ALPHA)
        if anomaly_y is not None:
            output = self._blend_polygon_fill(
                output,
                polygon_points,
                RAIL_ANOMALY_COLOR,
                RAIL_ANOMALY_FILL_ALPHA,
                max_y=anomaly_y,
            )

        for points in polygon_points:
            self.cv2.polylines(output, [points], True, RAIL_COLOR, 3)

        frame_height, frame_width = output.shape[:2]
        if anomaly_y is not None:
            clipped_bottom = max(0.0, min(float(frame_height - 1), anomaly_y))
            anomaly_clip = self.Polygon(
                [
                    (0.0, 0.0),
                    (float(frame_width - 1), 0.0),
                    (float(frame_width - 1), clipped_bottom),
                    (0.0, clipped_bottom),
                ]
            )
            for polygon in polygons:
                clipped = polygon.intersection(anomaly_clip)
                for part in self._polygon_parts(clipped):
                    points = self.np.array(part.exterior.coords, dtype=self.np.int32)
                    self.cv2.polylines(output, [points], True, RAIL_ANOMALY_COLOR, 3)

        for x_min, y_min, x_max, y_max, confidence, class_name, _score, intersects in boxes:
            color = ON_RAIL_COLOR if intersects else OFF_RAIL_COLOR
            box_start = (max(0, int(x_min)), max(0, int(y_min)))
            box_end = (min(frame_width - 1, int(x_max)), min(frame_height - 1, int(y_max)))
            self.cv2.rectangle(output, box_start, box_end, color, 3)

            label = f"{class_name} {confidence * 100:.0f}%"
            font = self.cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            padding = 5
            (text_width, text_height), baseline = self.cv2.getTextSize(label, font, font_scale, thickness)
            label_width = text_width + 2 * padding
            label_height = text_height + baseline + 2 * padding
            label_left = min(box_start[0], max(0, frame_width - label_width))
            label_bottom = box_start[1] if box_start[1] >= label_height else min(frame_height - 1, box_start[1] + label_height)
            label_top = max(0, label_bottom - label_height)
            self.cv2.rectangle(
                output,
                (label_left, label_top),
                (min(frame_width - 1, label_left + label_width), label_bottom),
                color,
                -1,
            )
            self.cv2.putText(
                output,
                label,
                (label_left + padding, label_top + padding + text_height),
                font,
                font_scale,
                LABEL_TEXT_COLOR,
                thickness,
            )
        if self.config.debug_display:
            for x, extent in sample_points:
                self.cv2.line(output, (int(x), output.shape[0] - 1), (int(x), int(output.shape[0] - extent)), (255, 0, 255), 1)
        return output

    def _blend_polygon_fill(
        self,
        output: Any,
        polygon_points: list[Any],
        color: tuple[int, int, int],
        alpha: float,
        max_y: float | None = None,
    ) -> Any:
        """Blend polygon fill, optionally only beyond an anomaly toward the horizon."""
        if not polygon_points:
            return output
        if isinstance(output, getattr(self.np, "ndarray", ())):
            all_points = self.np.concatenate(polygon_points, axis=0)
            frame_height, frame_width = output.shape[:2]
            left = max(0, int(all_points[:, 0].min()))
            right = min(frame_width, int(all_points[:, 0].max()) + 1)
            top = max(0, int(all_points[:, 1].min()))
            bottom = min(frame_height, int(all_points[:, 1].max()) + 1)
            if max_y is not None:
                bottom = min(bottom, max(0, min(frame_height, int(max_y) + 1)))
            if left >= right or top >= bottom:
                return output
            region = output[top:bottom, left:right]
            overlay = region.copy()
            offset = self.np.array([left, top], dtype=self.np.int32)
            for points in polygon_points:
                self.cv2.fillPoly(overlay, [points - offset], color)
            output[top:bottom, left:right] = self.cv2.addWeighted(
                overlay,
                alpha,
                region,
                1.0 - alpha,
                0,
            )
            return output

        overlay = output.copy()
        for points in polygon_points:
            self.cv2.fillPoly(overlay, [points], color)
        return self.cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    def _pace_output(self) -> None:
        """Rate-limit publishing without ever replaying an old frame."""
        period_ns = max(1, round(1_000_000_000 / self.config.output_fps))
        current = now_ns()
        deadline = getattr(self, "next_output_ns", None)
        if deadline is None or current - deadline > period_ns:
            deadline = current
        if deadline > current:
            time.sleep((deadline - current) / 1_000_000_000)
        self.next_output_ns = deadline + period_ns

    def _complete_inference(
        self,
        packet: FramePacket,
        dropped: int,
        result: Any,
        inference_start: int,
        inference_end: int,
    ) -> dict[str, Any]:
        frame = packet.frame
        height, width = frame.shape[:2]
        postprocess_start = now_ns()
        rail_confidence = self.extract_rail_confidence(result)
        detected_polygons = self.extract_rail_polygons(result, frame.shape)
        self.update_rail_polygons(detected_polygons)

        rail_score = 0.0
        effective_height = 0.0
        sample_points: list[tuple[float, float]] = []
        polygon_scores = [
            compute_rail_polygon_score(polygon, height, width, self.cv2, self.np, self.Polygon)
            for polygon in self.cached_polygons
        ]
        if polygon_scores:
            rail_score, effective_height, sample_points = max(
                polygon_scores,
                key=lambda value: (value[0], -value[1]),
            )
        rail_anomaly = rail_score > 0.0 and effective_height >= 30.0
        rail_anomaly_y = (
            float(max(0.0, min(height - 1.0, height - effective_height)))
            if rail_anomaly
            else None
        )
        detections = self.extract_objects(result)
        intersects, boxes, danger_scores, object_score = self.tracker.update(
            detections,
            self.cached_polygons,
            height,
            timestamp_ns=postprocess_start,
        )
        overall_score = object_score if effective_height < 30 else max(object_score, rail_score)
        annotated = self.draw(
            frame,
            self.cached_polygons,
            boxes,
            sample_points,
            anomaly_y=rail_anomaly_y,
        )
        postprocess_end = now_ns()

        self._pace_output()
        encode_start = now_ns()
        self.stream_writer.write(annotated)
        encode_end = now_ns()
        fps = 0.0 if self.last_output_ns is None else 1.0 / max((encode_end - self.last_output_ns) / 1e9, 1e-9)
        self.last_output_ns = encode_end
        self.total_dropped_frames += dropped
        if self.processed_frames % max(1, round(self.config.output_fps)) == 0:
            self.gpu_utilization_percent = self.runtime.gpu_utilization_percent()

        metrics = {
            "frame_id": packet.frame_id,
            "source_ts_ns": packet.source_ts_ns,
            "receive_ts_ns": packet.receive_ts_ns,
            "inference_start_ts_ns": inference_start,
            "inference_end_ts_ns": inference_end,
            "output_ts_ns": encode_end,
            "decode_ms": packet.decode_ms,
            "queue_ms": delta_ms(packet.receive_ts_ns, inference_start),
            "inference_ms": delta_ms(inference_start, inference_end),
            "postprocess_ms": delta_ms(postprocess_start, postprocess_end),
            "publish_enqueue_ms": delta_ms(encode_start, encode_end),
            "encode_ms": self.stream_writer.last_encode_ms,
            "pipeline_ms": delta_ms(packet.receive_ts_ns, encode_end),
            "source_to_output_ms": delta_ms(packet.source_ts_ns, encode_end),
            "dropped_frames": dropped,
            "gpu_utilization_percent": self.gpu_utilization_percent,
        }
        self.latency_logger.log(metrics)
        self.recording.write(
            frame,
            annotated,
            {
                "frame_id": packet.frame_id,
                "overall_score": overall_score,
                "rail_score": rail_score,
                "rail_anomaly": rail_anomaly,
                "rail_anomaly_y": rail_anomaly_y,
                "rail_confidence": rail_confidence,
                "rail_mask_stale_frames": self.rail_mask_stale_frames,
                "danger_scores": danger_scores,
                "boxes": boxes,
                "metrics": metrics,
            },
        )
        return {
            "frame_id": packet.frame_id,
            "annotated": annotated,
            "score": overall_score,
            "rail_score": rail_score,
            "rail_anomaly": rail_anomaly,
            "rail_anomaly_y": rail_anomaly_y,
            "rail_confidence": rail_confidence,
            "rail_mask_stale_frames": self.rail_mask_stale_frames,
            "fps": fps,
            "intersects": intersects,
            "metrics": metrics,
        }

    def _infer(self, packet: FramePacket, dropped: int) -> dict[str, Any]:
        inference_start = now_ns()
        try:
            result = self.model.predict(packet.frame, self.config, self.runtime)[0]
        except Exception as exc:
            raise ModelInferenceError(f"unified model inference failed: {exc}") from exc
        return self._complete_inference(
            packet,
            dropped,
            result,
            inference_start,
            now_ns(),
        )

    def _publish_result(self, result: dict[str, Any]) -> bool:
        dropped = int(result["metrics"]["dropped_frames"])
        self.processed_frames += 1
        with self.results_lock:
            self.latest_frame = result["annotated"]
            self.status.update(
                {
                    "ready": True,
                    "frame_id": result["frame_id"],
                    "score": result["score"],
                    "rail_score": result["rail_score"],
                    "rail_anomaly": result["rail_anomaly"],
                    "rail_anomaly_y": result["rail_anomaly_y"],
                    "rail_confidence": result["rail_confidence"],
                    "rail_mask_stale_frames": result["rail_mask_stale_frames"],
                    "fps": result["fps"],
                    "dropped_frames": self.total_dropped_frames,
                    "encoder_dropped_frames": self.stream_writer.dropped_frames,
                    "webrtc_reconnects": self.stream_writer.reconnects,
                    "webrtc_ready": self.stream_writer.ready,
                    "intersects_rail": result["intersects"],
                    "latency": result["metrics"],
                    "input_decoder": self.reader.decode_backend,
                    "decoder_fallback_reason": self.reader.fallback_reason,
                    "webrtc_encoder": self.stream_writer.backend,
                    "encoder_fallback_reason": self.stream_writer.fallback_reason,
                    "effective_precision": self.model.effective_precision,
                    "gpu_utilization_percent": self.gpu_utilization_percent,
                    **self.runtime.state.to_dict(),
                }
            )
        log_time = now_ns()
        if self.processed_frames == 1 or log_time - getattr(self, "last_frame_log_ns", 0) >= 1_000_000_000:
            frame_id = result["frame_id"]
            fps = result["fps"]
            score = result["score"]
            pipeline_ms = result["metrics"]["pipeline_ms"]
            print(
                f"frame={frame_id} fps={fps:.2f} score={score:.1f} "
                f"pipeline_ms={pipeline_ms:.1f} dropped={dropped}",
                flush=True,
            )
            self.last_frame_log_ns = log_time
        return True

    def process(self, packet: FramePacket, dropped: int) -> bool:
        try:
            result = self._infer(packet, dropped)
        except ModelInferenceError as exc:
            if self.runtime.can_runtime_fallback() and self.config.model_format != "tensorrt":
                self.fallback_to_cpu(f"inference failed: {exc}")
                result = self._infer(packet, dropped)
            else:
                raise
        return self._publish_result(result)

    def _prepare_staged(self, packet: FramePacket, dropped: int) -> tuple[FramePacket, int, Any, int]:
        inference_start = now_ns()
        try:
            image = self.model.preprocess(packet.frame)
        except Exception as exc:
            raise ModelInferenceError(f"unified model preprocessing failed: {exc}") from exc
        return packet, dropped, image, inference_start

    def _run_gpu_stage(
        self,
        prepared: tuple[FramePacket, int, Any, int],
    ) -> tuple[FramePacket, int, Any, int, int]:
        packet, dropped, image, inference_start = prepared
        try:
            result = self.model.predict_preprocessed(packet.frame, image)[0]
        except Exception as exc:
            raise ModelInferenceError(f"unified model inference failed: {exc}") from exc
        return packet, dropped, result, inference_start, now_ns()

    def _complete_staged(
        self,
        staged: tuple[FramePacket, int, Any, int, int],
    ) -> bool:
        packet, dropped, result, inference_start, inference_end = staged
        completed = self._complete_inference(
            packet,
            dropped,
            result,
            inference_start,
            inference_end,
        )
        return self._publish_result(completed)

    def _run_serial(self) -> None:
        while not self.stop_event.is_set():
            packet_info = self.buffer.get_latest()
            if packet_info is None:
                if self.buffer.stopped:
                    break
                continue
            packet, dropped = packet_info
            if not self.process(packet, dropped):
                break
            if self.config.max_frames and self.processed_frames >= self.config.max_frames:
                break

    def _run_staged(self) -> None:
        """Overlap frame preprocessing, serialized TensorRT and CPU rendering."""
        gpu_future: concurrent.futures.Future[Any] | None = None
        post_future: concurrent.futures.Future[Any] | None = None
        submitted = 0
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rail-trt") as gpu_pool,
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rail-post") as post_pool,
        ):
            while not self.stop_event.is_set():
                if self.config.max_frames and submitted >= self.config.max_frames:
                    break
                packet_info = self.buffer.get_latest()
                if packet_info is None:
                    if self.buffer.stopped:
                        break
                    continue
                prepared = self._prepare_staged(*packet_info)
                if gpu_future is not None:
                    staged = gpu_future.result()
                    if post_future is not None:
                        post_future.result()
                    post_future = post_pool.submit(self._complete_staged, staged)
                gpu_future = gpu_pool.submit(self._run_gpu_stage, prepared)
                submitted += 1

            if gpu_future is not None:
                staged = gpu_future.result()
                if post_future is not None:
                    post_future.result()
                post_future = post_pool.submit(self._complete_staged, staged)
            if post_future is not None:
                post_future.result()

    def run(self) -> int:
        self.start()
        exit_code = 0
        try:
            staged = self.config.platform == "jetson" and self.model.supports_staged_inference
            if staged:
                print("Pipeline scheduling: staged TensorRT -> CPU mask/render, capped at output FPS", flush=True)
                self._run_staged()
            else:
                self._run_serial()
        except Exception as exc:
            exit_code = 1
            with self.results_lock:
                self.status["error"] = str(exc)
                self.status["ready"] = False
            print(f"Pipeline failed: {exc}", flush=True)
        finally:
            self.close()
        if self.reader.error and self.processed_frames == 0:
            print(self.reader.error, flush=True)
            return 1
        return exit_code

    def status_snapshot(self) -> dict[str, Any]:
        with self.results_lock:
            snapshot = dict(self.status)
        snapshot["input_error"] = self.reader.error
        return snapshot
