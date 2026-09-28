"""HTTP health, status, score and model information API."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _release() -> str:
    release_file = Path(os.getenv("RAIL_AI_RELEASE_FILE", "/app/RELEASE"))
    try:
        return release_file.read_text(encoding="utf-8").strip()
    except OSError:
        return os.getenv("RAIL_AI_RELEASE", "development")


def create_app(pipeline: Any):
    from flask import Flask, Response, jsonify  # type: ignore

    app = Flask(__name__)
    model_name = Path(str(pipeline.config.model)).name

    @app.after_request
    def common_headers(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.get("/health/live")
    def health_live():
        status = pipeline.status_snapshot()
        return jsonify({"live": status["live"]}), (200 if status["live"] else 503)

    @app.get("/health/ready")
    def health_ready():
        status = pipeline.status_snapshot()
        ready = bool(
            status["live"]
            and status["ready"]
            and (not status.get("webrtc_enabled") or status.get("webrtc_ready"))
        )
        return jsonify({"ready": ready}), (200 if ready else 503)

    @app.get("/api/v1/status")
    def api_status():
        return jsonify(pipeline.status_snapshot())

    @app.get("/api/v1/info")
    def api_info():
        status = pipeline.status_snapshot()
        return jsonify(
            {
                "name": "ReaktRailAi",
                "release": _release(),
                "models": [
                    {
                        "name": model_name,
                        "role": "Schienensegmentierung und Objekterkennung",
                        "technology": "Ultralytics YOLO-Segmentierung",
                    },
                ],
                "function": (
                    "Die Anwendung segmentiert den Schienenbereich, erkennt und verfolgt "
                    "Hindernisse und berechnet daraus einen Gefahrenscore."
                ),
                "stream": {
                    "input": "RTSP",
                    "output": "WHEP/WebRTC",
                    "whep_path": "/processed/whep",
                    "codec": "H.264",
                },
                "runtime": {
                    "platform": status.get("platform"),
                    "device": status.get("effective_device"),
                    "provider": status.get("provider"),
                    "format": status.get("effective_format"),
                    "precision": status.get("effective_precision"),
                },
            }
        )

    @app.get("/api/v1/score")
    def api_score():
        status = pipeline.status_snapshot()
        return jsonify(
            {
                "frame_id": status["frame_id"],
                "score": status["score"],
                "rail_score": status["rail_score"],
                "intersects_rail": status.get("intersects_rail", False),
            }
        )

    def score_events(json_payload: bool):
        last_frame = -1
        while not pipeline.stop_event.is_set():
            status = pipeline.status_snapshot()
            if status["frame_id"] != last_frame:
                last_frame = status["frame_id"]
                payload = (
                    json.dumps(
                        {
                            "frame_id": status["frame_id"],
                            "score": status["score"],
                            "rail_score": status["rail_score"],
                        }
                    )
                    if json_payload
                    else str(status["score"])
                )
                yield f"id:{status['frame_id']}\ndata:{payload}\n\n"
            time.sleep(0.05)

    @app.get("/api/v1/events")
    def events():
        return Response(score_events(True), mimetype="text/event-stream")

    @app.get("/score_stream")
    def events_compat():
        return Response(score_events(False), mimetype="text/event-stream")

    @app.get("/score")
    def score_compat():
        return api_score()

    @app.get("/status")
    def status_compat():
        return api_status()

    return app
