from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
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
TRAINING_GENERATIONS_PATH = Path("data/prompt_suite_generations.json")
TRAINING_LOG_PATH = Path("logs/prompt_training.log")
ENABLE_MAX_DETAIL_UI_PANELS = str(os.getenv("ENABLE_MAX_DETAIL_UI_PANELS", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
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


@app.get("/training-runs")
async def training_runs_ui():
    page_path = UI_DIR / "training_runs.html"
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


def _load_generation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    generations = payload.get("generations", [])
    if not isinstance(generations, list):
        return []
    return [item for item in generations if isinstance(item, dict)]


def _group_records_by_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        run_id = metadata.get("run_id")
        if not run_id:
            continue
        grouped.setdefault(str(run_id), []).append(record)
    for run_id, run_records in grouped.items():
        run_records.sort(
            key=lambda item: (
                int((item.get("metadata") or {}).get("epoch", -1)),
                float(item.get("timestamp") or 0.0),
            )
        )
    return grouped


def _run_epoch_records(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epochs: list[dict[str, Any]] = []
    for item in run_records:
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and "epoch" in metadata:
            epochs.append(item)
    return epochs


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
            "feature_flags": {
                "ENABLE_MAX_DETAIL_UI_PANELS": ENABLE_MAX_DETAIL_UI_PANELS,
            },
        }
    return {
        "status": "ok",
        "tracking": status,
        "status_path": str(TRAINING_STATUS_PATH),
        "events_path": str(TRAINING_EVENTS_PATH),
        "log_path": str(TRAINING_LOG_PATH),
        "feature_flags": {
            "ENABLE_MAX_DETAIL_UI_PANELS": ENABLE_MAX_DETAIL_UI_PANELS,
        },
    }


@app.get("/training/events")
async def training_events(limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    return {
        "status": "ok",
        "events": _tail_jsonl(TRAINING_EVENTS_PATH, limit=limit),
        "events_path": str(TRAINING_EVENTS_PATH),
    }


@app.get("/training/runs")
async def training_runs() -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    runs: list[dict[str, Any]] = []
    for run_id, run_records in grouped.items():
        epoch_records = _run_epoch_records(run_records)
        timestamps = [float(item.get("timestamp") or 0.0) for item in run_records]
        last_epoch_metadata = (epoch_records[-1].get("metadata") if epoch_records else {}) or {}
        runs.append(
            {
                "run_id": run_id,
                "epochs_count": len(epoch_records),
                "start_timestamp": min(timestamps) if timestamps else None,
                "end_timestamp": max(timestamps) if timestamps else None,
                "latest_winner": last_epoch_metadata.get("winner"),
                "latest_improvement_delta": last_epoch_metadata.get("improvement_delta"),
                "events_archive_path": last_epoch_metadata.get("events_archive_path"),
            }
        )
    runs.sort(key=lambda item: float(item.get("end_timestamp") or 0.0), reverse=True)
    return {
        "status": "ok",
        "runs": runs,
        "runs_count": len(runs),
        "history_path": str(TRAINING_GENERATIONS_PATH),
    }


@app.get("/training/runs/{run_id}/summary")
async def training_run_summary(run_id: str) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    epochs = _run_epoch_records(run_records)
    if not epochs:
        return {"status": "error", "message": f"Run has no epoch records: {run_id}"}

    last_metadata = (epochs[-1].get("metadata") if epochs else {}) or {}
    improved_epochs = sum(
        1
        for item in epochs
        if float((item.get("metadata") or {}).get("improvement_delta") or 0.0) > 0.0
    )
    accepted_epochs = sum(
        1 for item in epochs if str(((item.get("metadata") or {}).get("winner") or "")) == "candidate"
    )
    net_delta = sum(float(((item.get("metadata") or {}).get("improvement_delta") or 0.0)) for item in epochs)

    return {
        "status": "ok",
        "run_id": run_id,
        "epochs_count": len(epochs),
        "improved_epochs": improved_epochs,
        "accepted_epochs": accepted_epochs,
        "net_improvement_delta": net_delta,
        "latest_epoch": int(last_metadata.get("epoch", -1)),
        "latest_winner": last_metadata.get("winner"),
        "latest_candidate_gate_failure_reasons": last_metadata.get("candidate_gate_failure_reasons", []),
        "events_archive_path": last_metadata.get("events_archive_path"),
    }


@app.get("/training/runs/{run_id}/epochs")
async def training_run_epochs(run_id: str) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    epochs = _run_epoch_records(run_records)
    epoch_rows: list[dict[str, Any]] = []
    for item in epochs:
        metadata = item.get("metadata") or {}
        epoch_rows.append(
            {
                "epoch": metadata.get("epoch"),
                "winner": metadata.get("winner"),
                "avg_score_a": metadata.get("avg_score_a"),
                "avg_score_b": metadata.get("avg_score_b"),
                "improvement_delta": metadata.get("improvement_delta"),
                "changed_keys": metadata.get("changed_keys", []),
                "candidate_gate_failure_reasons": metadata.get("candidate_gate_failure_reasons", []),
                "candidate_gate_results": metadata.get("candidate_gate_results", {}),
                "phase_a_eval_summary": metadata.get("phase_a_eval_summary", {}),
                "phase_b_eval_summary": metadata.get("phase_b_eval_summary", {}),
                "paired_phase_a_eval_summary": metadata.get("paired_phase_a_eval_summary", {}),
                "paired_phase_b_eval_summary": metadata.get("paired_phase_b_eval_summary", {}),
                "timeout_stats": metadata.get("timeout_stats", {}),
                "mutation_retry_summary": metadata.get("mutation_retry_summary", {}),
                "mutation_rejection_breakdown": metadata.get("mutation_rejection_breakdown", {}),
                "selection_stats": metadata.get("selection_stats", {}),
                "prompt_scoring_summary": metadata.get("prompt_scoring_summary", {}),
            }
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "epochs": epoch_rows,
        "epochs_count": len(epoch_rows),
    }


@app.get("/training/runs/{run_id}/epoch/{epoch}/cases")
async def training_run_epoch_cases(run_id: str, epoch: int) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    target_metadata: dict[str, Any] | None = None
    for item in _run_epoch_records(run_records):
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and int(metadata.get("epoch", -1)) == int(epoch):
            target_metadata = metadata
            break
    if target_metadata is None:
        return {"status": "error", "message": f"Epoch not found for run {run_id}: {epoch}"}

    scores_a = list(target_metadata.get("scores_a") or [])
    scores_b = list(target_metadata.get("scores_b") or [])
    train_case_ids = list(target_metadata.get("train_case_ids") or [])
    pair_count = min(len(scores_a), len(scores_b))
    cases: list[dict[str, Any]] = []
    for idx in range(pair_count):
        cases.append(
            {
                "index": idx + 1,
                "case_id": train_case_ids[idx] if idx < len(train_case_ids) else None,
                "score_a": scores_a[idx],
                "score_b": scores_b[idx],
                "delta": float(scores_b[idx]) - float(scores_a[idx]),
            }
        )
    return {
        "status": "ok",
        "run_id": run_id,
        "epoch": epoch,
        "cases": cases,
        "pair_count": pair_count,
    }


@app.get("/training/stream")
async def training_stream(poll_ms: int = Query(default=500, ge=200, le=5000)) -> StreamingResponse:
    def event_stream():
        position = 0
        heartbeat_interval = 3.0
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
                now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
                payload = {
                    "event_type": "heartbeat",
                    "timestamp": now_iso,
                    "timestamp_unix": round(now, 3),
                    "run_id": status.get("run_id"),
                    "state": status.get("state", "idle"),
                    "epoch_current": status.get("epoch_current", 0),
                    "epochs_total": status.get("epochs_total", 0),
                    "phase": status.get("phase", "idle"),
                    "step": status.get("step", "idle"),
                    "message": status.get("message", ""),
                    "events_count": status.get("events_count", 0),
                    "current_case": status.get("current_case"),
                    "active_call": status.get("active_call"),
                    "stalled_warning_count": status.get("stalled_warning_count", 0),
                    "elapsed_seconds": status.get("elapsed_seconds", 0.0),
                    "phase_elapsed_seconds": status.get("phase_elapsed_seconds", 0.0),
                    "metrics": status.get("metrics", {}),
                    "feature_flags": {
                        "ENABLE_MAX_DETAIL_UI_PANELS": ENABLE_MAX_DETAIL_UI_PANELS,
                    },
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
