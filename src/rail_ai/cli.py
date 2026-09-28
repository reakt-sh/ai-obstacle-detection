"""Command-line entry point."""

from __future__ import annotations

import signal
import threading

from .api import create_app
from .config import config_from_args
from .pipeline import RailPipeline


def main(argv: list[str] | None = None) -> int:
    try:
        config = config_from_args(argv)
        pipeline = RailPipeline(config)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Configuration/startup error: {exc}", flush=True)
        return 2

    print(
        f"ReaktRailAi starting: requested={config.requested_device}/{config.model_format}/{config.model_precision} "
        f"effective={pipeline.runtime.state.effective_device}/{pipeline.model.format}/"
        f"{pipeline.model.effective_precision} "
        f"input=RTSP output=WHEP/WebRTC",
        flush=True,
    )
    signal.signal(signal.SIGTERM, lambda _signum, _frame: pipeline.stop())
    if not config.api_enabled:
        return pipeline.run()

    worker_result = {"exit_code": 1}

    def run_pipeline() -> None:
        worker_result["exit_code"] = pipeline.run()

    worker = threading.Thread(target=run_pipeline, daemon=True)
    worker.start()
    app = create_app(pipeline)
    from werkzeug.serving import make_server  # type: ignore

    server = make_server(config.api_host, config.api_port, app, threaded=True)
    server.timeout = 0.5
    print(f"API listening on http://{config.api_host}:{config.api_port}", flush=True)
    try:
        while worker.is_alive():
            server.handle_request()
    except KeyboardInterrupt:
        pipeline.stop()
    finally:
        server.server_close()
        # Ask the pipeline thread to stop and let it close its own resources.
        # Closing the publisher here could interrupt an active frame write and
        # leave the publisher in an invalid state.
        pipeline.stop()
        worker.join(timeout=10)
        if worker.is_alive():
            print("Pipeline thread did not stop within 10 seconds", flush=True)
            worker_result["exit_code"] = 1
    return worker_result["exit_code"]
