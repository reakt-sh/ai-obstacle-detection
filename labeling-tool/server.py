#!/usr/bin/env python3
"""Small web server for reviewing ReaktRailAi segmentation labels."""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import threading
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
except ImportError:  # The editor still starts; only polygon merging is unavailable.
    Polygon = None
    box = None
    unary_union = None


ROOT = Path(__file__).resolve().parent
MAX_MERGE_GAP_PIXELS = 2.0


def _polygon_parts(geometry: object) -> list[object]:
    if getattr(geometry, "is_empty", True):
        return []
    if getattr(geometry, "geom_type", "") == "Polygon":
        return [geometry]
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if getattr(part, "geom_type", "") == "Polygon" and part.area > 0
    ]


def merge_polygon_points(
    polygons: object, width: object, height: object
) -> tuple[list[list[float]], bool]:
    if Polygon is None or box is None or unary_union is None:
        raise RuntimeError(
            "Für das Zusammenführen fehlt Shapely. Bitte die Abhängigkeiten aus labeling-tool/requirements.txt installieren."
        )
    width = float(width)
    height = float(height)
    if width <= 0 or height <= 0:
        raise ValueError("Ungültige Bildgröße")
    if not isinstance(polygons, list) or len(polygons) != 2:
        raise ValueError("Es müssen genau zwei Masken übergeben werden")

    geometries = []
    for index, points in enumerate(polygons, start=1):
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Maske {index} hat weniger als drei Punkte")
        try:
            geometry = Polygon([(float(point[0]), float(point[1])) for point in points])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Maske {index} enthält ungültige Punkte") from exc
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        parts = _polygon_parts(geometry)
        if not parts:
            raise ValueError(f"Maske {index} besitzt keine gültige Fläche")
        geometries.append(unary_union(parts))

    gap = float(geometries[0].distance(geometries[1]))
    snapped = 0 < gap <= MAX_MERGE_GAP_PIXELS
    if snapped:
        radius = gap / 2.0 + 0.05
        merged = unary_union(
            [geometry.buffer(radius, join_style="mitre") for geometry in geometries]
        ).buffer(-radius, join_style="mitre")
    else:
        merged = unary_union(geometries)
    merged = merged.intersection(box(0.0, 0.0, width, height))
    parts = _polygon_parts(merged)
    if len(parts) != 1:
        raise ValueError(
            f"Die Masken sind nicht verbunden (Abstand {gap:.1f} px). "
            "Bitte einen Eckpunkt so verschieben, dass sich die Masken berühren oder überlappen."
        )

    points = [
        [round(float(x), 6), round(float(y), 6)]
        for x, y in list(parts[0].exterior.coords)[:-1]
    ]
    if len(points) < 3:
        raise ValueError("Beim Zusammenführen ist keine gültige Maske entstanden")
    return points, snapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class ReviewStore:
    def __init__(self, dataset: Path) -> None:
        self.dataset = dataset.resolve()
        self.manifest_path = self.dataset / "candidates.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing dataset manifest: {self.manifest_path}")
        self.lock = threading.RLock()

    def manifest(self) -> dict:
        with self.lock:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def annotation_path(self, candidate_id: str) -> Path:
        if not candidate_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in candidate_id):
            raise ValueError("Invalid candidate id")
        return self.dataset / "annotations" / f"{candidate_id}.json"

    def annotation(self, candidate_id: str) -> dict:
        path = self.annotation_path(candidate_id)
        if not path.is_file():
            raise FileNotFoundError(candidate_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, candidate_id: str, annotation: dict) -> dict:
        if annotation.get("id") != candidate_id:
            raise ValueError("Annotation id does not match route")
        self._validate(annotation)
        with self.lock:
            manifest = self.manifest()
            candidate = next(
                (item for item in manifest.get("candidates", []) if item.get("id") == candidate_id),
                None,
            )
            if (
                candidate is None
                or annotation.get("source_video") != manifest.get("video")
                or annotation.get("source_frame") != candidate.get("source_frame")
            ):
                raise ValueError(
                    "Diese Anmerkung gehört nicht zum aktuell geöffneten Datensatz. "
                    "Bitte die Seite neu laden; der alte Labelstand bleibt erhalten."
                )
            atomic_json(self.annotation_path(candidate_id), annotation)
            self._write_yolo_label(annotation)
            for candidate in manifest.get("candidates", []):
                if candidate.get("id") == candidate_id:
                    candidate["reviewed"] = bool(annotation.get("reviewed"))
                    candidate["excluded"] = bool(annotation.get("excluded"))
                    candidate["instance_count"] = len(annotation.get("instances", []))
                    break
            atomic_json(self.manifest_path, manifest)
        return annotation

    @staticmethod
    def _validate(annotation: dict) -> None:
        width = int(annotation.get("width", 0))
        height = int(annotation.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")
        seen = set()
        for instance in annotation.get("instances", []):
            instance_id = str(instance.get("id", ""))
            if not instance_id or instance_id in seen:
                raise ValueError("Instance ids must be unique")
            seen.add(instance_id)
            class_id = int(instance.get("class_id", -1))
            if class_id < 0:
                raise ValueError("Invalid class id")
            points = instance.get("points", [])
            if len(points) < 3:
                raise ValueError(f"Instance {instance_id} has fewer than three points")
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"Invalid point in {instance_id}")
                x, y = float(point[0]), float(point[1])
                if not (-width <= x <= width * 2 and -height <= y <= height * 2):
                    raise ValueError(f"Point far outside image in {instance_id}")

    def _write_yolo_label(self, annotation: dict) -> None:
        labels_dir = self.dataset / "labels"
        labels_dir.mkdir(exist_ok=True)
        label_path = labels_dir / f'{annotation["id"]}.txt'
        if not annotation.get("reviewed") or annotation.get("excluded"):
            label_path.unlink(missing_ok=True)
            return
        width = float(annotation["width"])
        height = float(annotation["height"])
        lines = []
        for instance in annotation.get("instances", []):
            coordinates = []
            for x, y in instance["points"]:
                coordinates.extend(
                    (
                        min(1.0, max(0.0, float(x) / width)),
                        min(1.0, max(0.0, float(y) / height)),
                    )
                )
            lines.append(
                " ".join(
                    [str(int(instance["class_id"]))]
                    + [f"{value:.6f}" for value in coordinates]
                )
            )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def export_zip(self) -> bytes:
        manifest = self.manifest()
        classes = manifest.get("classes", [])
        reviewed = [
            item for item in manifest.get("candidates", [])
            if item.get("reviewed") and not item.get("excluded")
        ]
        data_yaml = ["path: .", "train: images", "val: images", f"nc: {len(classes)}", "names:"]
        for item in classes:
            data_yaml.append(f'  {int(item["id"])}: {json.dumps(item["name"], ensure_ascii=False)}')
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.yaml", "\n".join(data_yaml) + "\n")
            archive.writestr("candidates.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            for candidate in reviewed:
                candidate_id = candidate["id"]
                annotation = self.annotation(candidate_id)
                image_path = self.dataset / annotation["image"]
                label_path = self.dataset / "labels" / f"{candidate_id}.txt"
                archive.write(image_path, f"images/{image_path.name}")
                if label_path.is_file():
                    archive.write(label_path, f"labels/{label_path.name}")
                archive.writestr(
                    f"annotations/{candidate_id}.json",
                    json.dumps(annotation, indent=2, ensure_ascii=False) + "\n",
                )
        return memory.getvalue()


class Handler(SimpleHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format_string % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/manifest":
                return self._json(self.store.manifest())
            if route.startswith("/api/annotation/"):
                candidate_id = unquote(route.removeprefix("/api/annotation/"))
                return self._json(self.store.annotation(candidate_id))
            if route == "/api/export":
                payload = self.store.export_zip()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="reaktrailai-reviewed-labels.zip"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route.startswith("/data/images/"):
                name = unquote(route.removeprefix("/data/images/"))
                return self._file(self.store.dataset / "images" / Path(name).name)
            if route == "/" or route == "/index.html":
                return self._file(ROOT / "index.html")
            if route.startswith("/static/"):
                name = Path(unquote(route.removeprefix("/static/"))).name
                return self._file(ROOT / name)
            self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route != "/api/merge-polygons" and not route.startswith("/api/annotation/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/merge-polygons":
                if not isinstance(payload, dict):
                    raise ValueError("Ungültige Anfrage")
                points, snapped = merge_polygon_points(
                    payload.get("polygons"), payload.get("width"), payload.get("height")
                )
                self._json({"ok": True, "points": points, "snapped": snapped})
                return
            candidate_id = unquote(route.removeprefix("/api/annotation/"))
            saved = self.store.save(candidate_id, payload)
            self._json({"ok": True, "annotation": saved})
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        cache_control = (
            "no-store"
            if path.suffix.lower() in {".html", ".js", ".css"}
            else "public, max-age=3600"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    args = parse_args()
    store = ReviewStore(args.dataset)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ReaktRailAi annotation tool: http://127.0.0.1:{args.port}", flush=True)
    print(f"Dataset: {store.dataset}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
