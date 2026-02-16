from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from pipeline import Pipeline
    from utility import load_prompts
except Exception:  # pragma: no cover
    from src.pipeline import Pipeline
    from src.utility import load_prompts


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return str(obj)
        return super().default(obj)


pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UI_DIR = Path(__file__).resolve().parent / "ui"
TRAINING_STATUS_PATH = Path("data/training_status.json")
TRAINING_EVENTS_PATH = Path("data/training_events.jsonl")
TRAINING_LOG_PATH = Path("logs/prompt_training.log")
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


@app.get("/")
async def root():
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "error", "message": f"Missing UI file: {index_path}"}


@app.get("/training")
async def training_ui():
    page_path = UI_DIR / "training.html"
    if page_path.exists():
        return FileResponse(str(page_path))
    return {"status": "error", "message": f"Missing training UI file: {page_path}"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []

    items: deque[str] = deque(maxlen=limit)
    try:
        with path.open() as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    items.append(line)
    except Exception:
        return []

    parsed: list[dict[str, Any]] = []
    for line in items:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                parsed.append(payload)
        except Exception:
            continue
    return parsed


@app.get("/training/status")
async def training_status() -> dict[str, Any]:
    status = _read_json_file(TRAINING_STATUS_PATH)
    if status is None:
        return {
            "status": "idle",
            "tracking": None,
            "status_path": str(TRAINING_STATUS_PATH),
            "events_path": str(TRAINING_EVENTS_PATH),
            "log_path": str(TRAINING_LOG_PATH),
        }
    return {
        "status": "ok",
        "tracking": status,
        "status_path": str(TRAINING_STATUS_PATH),
        "events_path": str(TRAINING_EVENTS_PATH),
        "log_path": str(TRAINING_LOG_PATH),
    }


@app.get("/training/events")
async def training_events(limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    return {
        "status": "ok",
        "events": _tail_jsonl(TRAINING_EVENTS_PATH, limit=limit),
        "events_path": str(TRAINING_EVENTS_PATH),
    }


@app.get("/training/stream")
async def training_stream(poll_ms: int = Query(default=1000, ge=200, le=5000)) -> StreamingResponse:
    def event_stream():
        position = 0
        heartbeat_interval = 5.0
        last_heartbeat = 0.0

        while True:
            emitted = False
            if TRAINING_EVENTS_PATH.exists():
                try:
                    current_size = TRAINING_EVENTS_PATH.stat().st_size
                    if current_size < position:
                        # Training run reset the event file; restart tail from the beginning.
                        position = 0
                    with TRAINING_EVENTS_PATH.open() as handle:
                        handle.seek(position)
                        while True:
                            raw = handle.readline()
                            if raw == "":
                                break
                            line = raw.strip()
                            position = handle.tell()
                            if not line:
                                continue
                            emitted = True
                            yield f"data: {line}\n\n"
                except Exception as exc:
                    payload = {"event_type": "stream_error", "message": str(exc)}
                    yield f"data: {json.dumps(payload, cls=JSONEncoder)}\n\n"

            now = time.time()
            if not emitted and (now - last_heartbeat) >= heartbeat_interval:
                status = _read_json_file(TRAINING_STATUS_PATH) or {}
                payload = {
                    "event_type": "heartbeat",
                    "timestamp": now,
                    "run_id": status.get("run_id"),
                    "state": status.get("state", "idle"),
                    "epoch_current": status.get("epoch_current", 0),
                    "epochs_total": status.get("epochs_total", 0),
                    "phase": status.get("phase", "idle"),
                    "step": status.get("step", "idle"),
                    "message": status.get("message", ""),
                    "events_count": status.get("events_count", 0),
                    "current_case": status.get("current_case"),
                    "elapsed_seconds": status.get("elapsed_seconds", 0.0),
                    "phase_elapsed_seconds": status.get("phase_elapsed_seconds", 0.0),
                    "metrics": status.get("metrics", {}),
                }
                yield f"data: {json.dumps(payload, cls=JSONEncoder)}\n\n"
                last_heartbeat = now

            time.sleep(max(0.2, poll_ms / 1000.0))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reload-prompts")
async def reload_prompts() -> dict[str, Any]:
    try:
        loaded = load_prompts("data/prompts.json")
        prompt_content = loaded.get("content", {}) if isinstance(loaded, dict) else {}
        return {
            "status": "ok",
            "prompt_count": len(prompt_content),
            "prompt_ids": sorted(prompt_content.keys()),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@app.post("/run")
async def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    prompt = str(request.get("prompt", "")).strip()
    thinking_level = str(request.get("thinking_level", "med-synth"))
    if not prompt:
        return {"status": "error", "message": "Prompt cannot be empty"}

    if pipeline is None:
        return {"status": "error", "message": "Pipeline is not initialized"}

    result = pipeline.run(prompt=prompt, thinking_level=thinking_level)  # type: ignore[arg-type]
    return {
        "status": "ok",
        "result": result,
    }


@app.post("/stream")
async def stream_pipeline(request: dict[str, Any]) -> StreamingResponse:
    prompt = str(request.get("prompt", "")).strip()
    thinking_level = str(request.get("thinking_level", "med-synth"))

    if not prompt:
        async def error_stream():
            payload = {
                "event_type": "error",
                "stage": "error",
                "payload": {"message": "Prompt cannot be empty"},
            }
            yield f"data: {json.dumps(payload, cls=JSONEncoder)}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    def event_stream():
        if pipeline is None:
            payload = {
                "event_type": "error",
                "stage": "error",
                "payload": {"message": "Pipeline is not initialized"},
            }
            yield f"data: {json.dumps(payload, cls=JSONEncoder)}\n\n"
            return

        try:
            for event in pipeline.run_stream(prompt=prompt, thinking_level=thinking_level):  # type: ignore[arg-type]
                yield f"data: {json.dumps(event, cls=JSONEncoder)}\n\n"
        except Exception as exc:
            payload = {
                "event_type": "error",
                "stage": "error",
                "payload": {"message": str(exc)},
            }
            yield f"data: {json.dumps(payload, cls=JSONEncoder)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
