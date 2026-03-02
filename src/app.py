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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


ENABLE_MAX_DETAIL_UI_PANELS = str(os.getenv("ENABLE_MAX_DETAIL_UI_PANELS", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_TRAINING_V3_API = _env_flag("ENABLE_TRAINING_V3_API", True)
ENABLE_TRAINING_TRUTH_FLAGS = _env_flag("ENABLE_TRAINING_TRUTH_FLAGS", True)
ENABLE_TRAINING_UI_V3 = _env_flag("ENABLE_TRAINING_UI_V3", True)
ENABLE_RUNS_UI_V3 = _env_flag("ENABLE_RUNS_UI_V3", True)
ENABLE_PROMPT_INDEX_V3 = _env_flag("ENABLE_PROMPT_INDEX_V3", True)
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


def _find_epoch_metadata(run_records: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    for item in _run_epoch_records(run_records):
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and int(metadata.get("epoch", -1)) == int(epoch):
            return metadata
    return None


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _gate_state(
    gate_payload: dict[str, Any] | None,
    *,
    candidate_exists: bool,
    allow_without_candidate: bool = False,
) -> str:
    if not isinstance(gate_payload, dict):
        return "not_applicable"
    if (not candidate_exists) and (not allow_without_candidate):
        return "not_applicable"
    passed = gate_payload.get("passed")
    if passed is True:
        return "pass"
    if passed is False:
        return "fail"
    return "not_applicable"


def _normalize_epoch_truth(metadata: dict[str, Any]) -> dict[str, Any]:
    legacy_inferred = False

    epoch_index = _int_or(metadata.get("epoch_index", metadata.get("epoch", -1)), -1)
    epoch_number = _int_or(metadata.get("epoch_number", epoch_index + 1), epoch_index + 1)

    candidate_exists_raw = metadata.get("candidate_exists")
    evaluation_mode_raw = metadata.get("evaluation_mode")
    if isinstance(candidate_exists_raw, bool):
        candidate_exists = bool(candidate_exists_raw)
    else:
        reasons = set(str(item) for item in (metadata.get("candidate_gate_failure_reasons") or []))
        mutated = list(metadata.get("mutated_block_ids") or [])
        if "no_candidate_changes" in reasons and not mutated:
            candidate_exists = False
        elif mutated:
            candidate_exists = True
        else:
            candidate_exists = False
        legacy_inferred = True

    if isinstance(evaluation_mode_raw, str) and evaluation_mode_raw.strip():
        evaluation_mode = str(evaluation_mode_raw).strip()
    else:
        reasons = set(str(item) for item in (metadata.get("candidate_gate_failure_reasons") or []))
        if "no_candidate_changes" in reasons:
            evaluation_mode = "skipped_no_changes"
        elif candidate_exists:
            evaluation_mode = "evaluated"
        else:
            evaluation_mode = "synthetic"
        legacy_inferred = True

    truth_state_raw = metadata.get("truth_state")
    if isinstance(truth_state_raw, str) and truth_state_raw.strip():
        truth_state = str(truth_state_raw).strip()
    else:
        if candidate_exists and evaluation_mode == "evaluated":
            truth_state = "complete"
        elif evaluation_mode in {"synthetic", "skipped_no_changes"}:
            truth_state = "synthetic"
        else:
            truth_state = "partial"
        legacy_inferred = True

    data_quality_flags = list(metadata.get("data_quality_flags") or [])
    if not data_quality_flags:
        if truth_state == "complete":
            data_quality_flags = ["complete"]
        elif truth_state == "synthetic":
            data_quality_flags = ["synthetic"]
        else:
            data_quality_flags = ["partial"]
        legacy_inferred = True
    if legacy_inferred:
        data_quality_flags.append("legacy_inferred")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique_flags: list[str] = []
    for flag in data_quality_flags:
        key = str(flag)
        if key in seen:
            continue
        seen.add(key)
        unique_flags.append(key)

    return {
        "epoch_index": epoch_index,
        "epoch_number": epoch_number,
        "candidate_exists": candidate_exists,
        "evaluation_mode": evaluation_mode,
        "truth_state": truth_state,
        "legacy_inferred": legacy_inferred,
        "data_quality_flags": unique_flags,
    }


def _normalize_epoch_v3(metadata: dict[str, Any]) -> dict[str, Any]:
    truth = _normalize_epoch_truth(metadata)
    gate_results = (metadata.get("candidate_gate_results") or {}) if isinstance(metadata, dict) else {}
    gates = gate_results.get("gates") or {}
    holdout_confirmation = ((metadata.get("selection_stats") or {}).get("holdout_confirmation")) or None

    quality_gate = gates.get("quality_gate") if isinstance(gates, dict) else None
    runtime_gate = gates.get("runtime_gate") if isinstance(gates, dict) else None
    stability_gate = gates.get("stability_gate") if isinstance(gates, dict) else None
    tail_risk_gate = gates.get("tail_risk_gate") if isinstance(gates, dict) else None

    gate_states = {
        "quality": _gate_state(quality_gate, candidate_exists=truth["candidate_exists"]),
        "runtime": _gate_state(runtime_gate, candidate_exists=truth["candidate_exists"]),
        "stability": _gate_state(stability_gate, candidate_exists=truth["candidate_exists"]),
        "tail_risk": _gate_state(tail_risk_gate, candidate_exists=truth["candidate_exists"]),
        "holdout": _gate_state(
            holdout_confirmation if isinstance(holdout_confirmation, dict) else None,
            candidate_exists=truth["candidate_exists"],
            allow_without_candidate=False,
        ),
    }

    rejection_reasons = list(metadata.get("rejection_reason_codes") or [])
    if not rejection_reasons and metadata.get("candidate_gate_failure_reasons"):
        rejection_reasons = [
            {
                "code": str(code),
                "severity": "high" if str(code) in {"quality_gate", "tail_risk_gate", "no_candidate_changes"} else "medium",
                "rank": idx + 1,
                "evidence": "candidate_gate_failure_reasons",
                "legacy_inferred": True,
            }
            for idx, code in enumerate(list(metadata.get("candidate_gate_failure_reasons") or []))
        ]

    scores_a = list(metadata.get("scores_a") or [])
    scores_b = list(metadata.get("scores_b") or [])
    paired_count = min(len(scores_a), len(scores_b))
    if not truth["candidate_exists"]:
        # Enforce truth invariant: candidate-only result should not be presented as full B eval.
        paired_count = 0

    return {
        "schema_version": "3.0",
        "source_schema_version": str(metadata.get("metadata_schema_version") or "legacy"),
        "epoch_index": truth["epoch_index"],
        "epoch_number": truth["epoch_number"],
        "candidate_exists": truth["candidate_exists"],
        "evaluation_mode": truth["evaluation_mode"],
        "truth_state": truth["truth_state"],
        "legacy_inferred": truth["legacy_inferred"],
        "data_quality_flags": truth["data_quality_flags"],
        "winner": metadata.get("winner"),
        "scores": {
            "avg_score_a": _float_or_none(metadata.get("avg_score_a")),
            "avg_score_b": _float_or_none(metadata.get("avg_score_b")) if truth["candidate_exists"] else None,
            "improvement_delta": _float_or_none(metadata.get("improvement_delta")) if truth["candidate_exists"] else None,
            "mean_delta": _float_or_none(metadata.get("mean_delta")) if truth["candidate_exists"] else None,
        },
        "paired_case_count": paired_count,
        "changed_keys": list(metadata.get("changed_keys") or []),
        "mutated_block_ids": list(metadata.get("mutated_block_ids") or []),
        "selection_blocked_reason": metadata.get("selection_blocked_reason"),
        "gates": {
            "quality": {
                "gate_state": gate_states["quality"],
                "payload": quality_gate,
            },
            "runtime": {
                "gate_state": gate_states["runtime"],
                "payload": runtime_gate,
            },
            "stability": {
                "gate_state": gate_states["stability"],
                "payload": stability_gate,
            },
            "tail_risk": {
                "gate_state": gate_states["tail_risk"],
                "payload": tail_risk_gate,
            },
            "holdout": {
                "gate_state": gate_states["holdout"],
                "payload": holdout_confirmation,
            },
        },
        "gate_rule_versions": metadata.get("gate_rule_versions", {}),
        "selection_stats": metadata.get("selection_stats", {}),
        "selection_diagnostics": metadata.get("selection_diagnostics", []),
        "rca_high_level_summary": metadata.get("rca_high_level_summary", {}),
        "prompt_scoring_summary": metadata.get("prompt_scoring_summary", {}),
        "mutation_accept_threshold": metadata.get("mutation_accept_threshold"),
        "holdout_winrate": metadata.get("holdout_winrate"),
        "event_schema_version": metadata.get("event_schema_version"),
        "metadata_schema_version": metadata.get("metadata_schema_version"),
        "rejection_reason_codes": rejection_reasons,
    }


def _find_epoch_metadata_by_number(run_records: list[dict[str, Any]], epoch_number: int) -> tuple[int, dict[str, Any] | None]:
    epoch_number_int = _int_or(epoch_number, 1)
    epoch_index = max(0, epoch_number_int - 1)
    metadata = _find_epoch_metadata(run_records, epoch_index)
    return epoch_index, metadata


def _build_run_summary_v3(run_id: str, run_records: list[dict[str, Any]]) -> dict[str, Any]:
    epoch_records = _run_epoch_records(run_records)
    epoch_v3 = [_normalize_epoch_v3((item.get("metadata") or {})) for item in epoch_records]
    timestamps = [float(item.get("timestamp") or 0.0) for item in run_records]

    accepted_epochs = sum(1 for row in epoch_v3 if str(row.get("winner") or "") == "candidate")
    improved_epochs = sum(
        1
        for row in epoch_v3
        if (row.get("scores") or {}).get("improvement_delta") is not None
        and float((row.get("scores") or {}).get("improvement_delta") or 0.0) > 0.0
    )
    net_delta = sum(
        float((row.get("scores") or {}).get("improvement_delta") or 0.0)
        for row in epoch_v3
        if (row.get("scores") or {}).get("improvement_delta") is not None
    )

    latest = epoch_v3[-1] if epoch_v3 else None
    return {
        "schema_version": "3.0",
        "source_schema_version": str((latest or {}).get("metadata_schema_version") or "legacy"),
        "run_id": str(run_id),
        "epoch_count": len(epoch_v3),
        "accepted_epochs": accepted_epochs,
        "improved_epochs": improved_epochs,
        "net_improvement_delta": net_delta,
        "start_timestamp": min(timestamps) if timestamps else None,
        "end_timestamp": max(timestamps) if timestamps else None,
        "latest_epoch_index": (latest or {}).get("epoch_index"),
        "latest_epoch_number": (latest or {}).get("epoch_number"),
        "latest_winner": (latest or {}).get("winner"),
        "latest_truth_state": (latest or {}).get("truth_state"),
        "latest_gate_states": {
            "quality": (((latest or {}).get("gates") or {}).get("quality") or {}).get("gate_state"),
            "runtime": (((latest or {}).get("gates") or {}).get("runtime") or {}).get("gate_state"),
            "stability": (((latest or {}).get("gates") or {}).get("stability") or {}).get("gate_state"),
            "tail_risk": (((latest or {}).get("gates") or {}).get("tail_risk") or {}).get("gate_state"),
            "holdout": (((latest or {}).get("gates") or {}).get("holdout") or {}).get("gate_state"),
        },
        "legacy_inferred": bool((latest or {}).get("legacy_inferred")),
        "data_quality_flags": list((latest or {}).get("data_quality_flags") or []),
    }


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
                "ENABLE_TRAINING_V3_API": ENABLE_TRAINING_V3_API,
                "ENABLE_TRAINING_TRUTH_FLAGS": ENABLE_TRAINING_TRUTH_FLAGS,
                "ENABLE_TRAINING_UI_V3": ENABLE_TRAINING_UI_V3,
                "ENABLE_RUNS_UI_V3": ENABLE_RUNS_UI_V3,
                "ENABLE_PROMPT_INDEX_V3": ENABLE_PROMPT_INDEX_V3,
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
            "ENABLE_TRAINING_V3_API": ENABLE_TRAINING_V3_API,
            "ENABLE_TRAINING_TRUTH_FLAGS": ENABLE_TRAINING_TRUTH_FLAGS,
            "ENABLE_TRAINING_UI_V3": ENABLE_TRAINING_UI_V3,
            "ENABLE_RUNS_UI_V3": ENABLE_RUNS_UI_V3,
            "ENABLE_PROMPT_INDEX_V3": ENABLE_PROMPT_INDEX_V3,
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
                "event_schema_version": last_epoch_metadata.get("event_schema_version"),
                "metadata_schema_version": last_epoch_metadata.get("metadata_schema_version"),
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
        "latest_gate_rule_versions": last_metadata.get("gate_rule_versions", {}),
        "latest_event_schema_version": last_metadata.get("event_schema_version"),
        "latest_metadata_schema_version": last_metadata.get("metadata_schema_version"),
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
                "selection_diagnostics": metadata.get("selection_diagnostics", []),
                "rca_high_level_summary": metadata.get("rca_high_level_summary", {}),
                "mutation_accept_threshold": metadata.get("mutation_accept_threshold"),
                "holdout_winrate": metadata.get("holdout_winrate"),
                "gate_rule_versions": metadata.get("gate_rule_versions", {}),
                "event_schema_version": metadata.get("event_schema_version"),
                "metadata_schema_version": metadata.get("metadata_schema_version"),
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

    target_metadata = _find_epoch_metadata(run_records, epoch)
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


@app.get("/training/runs/{run_id}/epoch/{epoch}/rca-themes")
async def training_run_epoch_rca_themes(run_id: str, epoch: int) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    target_metadata = _find_epoch_metadata(run_records, epoch)
    if target_metadata is None:
        return {"status": "error", "message": f"Epoch not found for run {run_id}: {epoch}"}

    high_level = target_metadata.get("rca_high_level_summary") or {}
    return {
        "status": "ok",
        "run_id": run_id,
        "epoch": epoch,
        "summary": high_level.get("summary"),
        "themes": high_level.get("themes", []),
        "model_used": high_level.get("model_used"),
    }


@app.get("/training/runs/{run_id}/epoch/{epoch}/mutation-decisions")
async def training_run_epoch_mutation_decisions(run_id: str, epoch: int) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    target_metadata = _find_epoch_metadata(run_records, epoch)
    if target_metadata is None:
        return {"status": "error", "message": f"Epoch not found for run {run_id}: {epoch}"}

    decisions = target_metadata.get("prompt_scoring") or []
    return {
        "status": "ok",
        "run_id": run_id,
        "epoch": epoch,
        "mutation_accept_threshold": target_metadata.get("mutation_accept_threshold"),
        "decisions": decisions,
        "count": len(decisions) if isinstance(decisions, list) else 0,
    }


@app.get("/training/runs/{run_id}/epoch/{epoch}/selection-diagnostics")
async def training_run_epoch_selection_diagnostics(run_id: str, epoch: int) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    target_metadata = _find_epoch_metadata(run_records, epoch)
    if target_metadata is None:
        return {"status": "error", "message": f"Epoch not found for run {run_id}: {epoch}"}

    return {
        "status": "ok",
        "run_id": run_id,
        "epoch": epoch,
        "selection_diagnostics": target_metadata.get("selection_diagnostics", []),
        "selection_stats": target_metadata.get("selection_stats", {}),
    }


@app.get("/training/runs/{run_id}/epoch/{epoch}/gate-timeline")
async def training_run_epoch_gate_timeline(run_id: str, epoch: int) -> dict[str, Any]:
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}"}

    target_metadata = _find_epoch_metadata(run_records, epoch)
    if target_metadata is None:
        return {"status": "error", "message": f"Epoch not found for run {run_id}: {epoch}"}

    gate_results = target_metadata.get("candidate_gate_results") or {}
    holdout_confirmation = (target_metadata.get("selection_stats") or {}).get("holdout_confirmation")
    return {
        "status": "ok",
        "run_id": run_id,
        "epoch": epoch,
        "gate_rule_versions": target_metadata.get("gate_rule_versions", {}),
        "candidate_gate_results": gate_results,
        "holdout_confirmation": holdout_confirmation,
        "winner": target_metadata.get("winner"),
    }


@app.get("/training/v3/status")
async def training_v3_status() -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    status = _read_json_file(TRAINING_STATUS_PATH)
    return {
        "status": "ok",
        "schema_version": "3.0",
        "source_schema_version": str((status or {}).get("metadata_schema_version") or "legacy"),
        "tracking": status,
        "truth_flags_enabled": ENABLE_TRAINING_TRUTH_FLAGS,
        "feature_flags": {
            "ENABLE_TRAINING_V3_API": ENABLE_TRAINING_V3_API,
            "ENABLE_TRAINING_TRUTH_FLAGS": ENABLE_TRAINING_TRUTH_FLAGS,
            "ENABLE_TRAINING_UI_V3": ENABLE_TRAINING_UI_V3,
            "ENABLE_RUNS_UI_V3": ENABLE_RUNS_UI_V3,
            "ENABLE_PROMPT_INDEX_V3": ENABLE_PROMPT_INDEX_V3,
        },
    }


@app.get("/training/v3/events")
async def training_v3_events(limit: int = Query(default=200, ge=1, le=4000)) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    events = _tail_jsonl(TRAINING_EVENTS_PATH, limit=limit)
    envelopes: list[dict[str, Any]] = []
    for item in events:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        envelopes.append(
            {
                "schema_version": "3.0",
                "source_schema_version": str(item.get("event_schema_version") or "legacy"),
                "event_type": item.get("event_type"),
                "timestamp": item.get("timestamp"),
                "epoch_current": item.get("epoch_current"),
                "run_id": item.get("run_id"),
                "truth_state": payload.get("truth_state"),
                "candidate_exists": payload.get("candidate_exists"),
                "evaluation_mode": payload.get("evaluation_mode"),
                "data_quality_flags": (
                    ["legacy_inferred"]
                    if str(item.get("event_schema_version") or "").strip() in {"", "legacy"}
                    else ["complete"]
                ),
                "payload": item,
            }
        )
    return {
        "status": "ok",
        "schema_version": "3.0",
        "events": envelopes,
        "events_count": len(envelopes),
        "events_path": str(TRAINING_EVENTS_PATH),
    }


@app.get("/training/v3/runs")
async def training_v3_runs() -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    runs = [_build_run_summary_v3(run_id, run_records) for run_id, run_records in grouped.items()]
    runs.sort(key=lambda item: float(item.get("end_timestamp") or 0.0), reverse=True)
    return {
        "status": "ok",
        "schema_version": "3.0",
        "runs": runs,
        "runs_count": len(runs),
        "history_path": str(TRAINING_GENERATIONS_PATH),
    }


@app.get("/training/v3/runs/{run_id}/summary")
async def training_v3_run_summary(run_id: str) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    summary = _build_run_summary_v3(str(run_id), run_records)
    summary["status"] = "ok"
    return summary


@app.get("/training/v3/runs/{run_id}/epochs")
async def training_v3_run_epochs(run_id: str) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    rows = [_normalize_epoch_v3((item.get("metadata") or {})) for item in _run_epoch_records(run_records)]
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epochs": rows,
        "epochs_count": len(rows),
    }


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}")
async def training_v3_run_epoch(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    row = _normalize_epoch_v3(metadata)
    row["status"] = "ok"
    row["run_id"] = str(run_id)
    return row


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}/cases")
async def training_v3_run_epoch_cases(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    epoch_v3 = _normalize_epoch_v3(metadata)
    if not epoch_v3.get("candidate_exists"):
        return {
            "status": "ok",
            "schema_version": "3.0",
            "run_id": str(run_id),
            "epoch_number": epoch_v3.get("epoch_number"),
            "epoch_index": epoch_v3.get("epoch_index"),
            "candidate_exists": False,
            "evaluation_mode": epoch_v3.get("evaluation_mode"),
            "truth_state": epoch_v3.get("truth_state"),
            "cases": [],
            "pair_count": 0,
            "note": "candidate_not_evaluated",
        }

    scores_a = list(metadata.get("scores_a") or [])
    scores_b = list(metadata.get("scores_b") or [])
    train_case_ids = list(metadata.get("train_case_ids") or [])
    pair_count = min(len(scores_a), len(scores_b))
    cases: list[dict[str, Any]] = []
    for idx in range(pair_count):
        score_a = _float_or_none(scores_a[idx]) if idx < len(scores_a) else None
        score_b = _float_or_none(scores_b[idx]) if idx < len(scores_b) else None
        if score_a is None or score_b is None:
            continue
        cases.append(
            {
                "schema_version": "3.0",
                "source_schema_version": str(metadata.get("metadata_schema_version") or "legacy"),
                "index": idx + 1,
                "case_id": train_case_ids[idx] if idx < len(train_case_ids) else None,
                "score_a": score_a,
                "score_b": score_b,
                "delta": float(score_b) - float(score_a),
                "candidate_exists": True,
                "evaluation_mode": epoch_v3.get("evaluation_mode"),
                "truth_state": epoch_v3.get("truth_state"),
            }
        )
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epoch_number": epoch_v3.get("epoch_number"),
        "epoch_index": epoch_v3.get("epoch_index"),
        "candidate_exists": True,
        "evaluation_mode": epoch_v3.get("evaluation_mode"),
        "truth_state": epoch_v3.get("truth_state"),
        "cases": cases,
        "pair_count": len(cases),
    }


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}/gates")
async def training_v3_run_epoch_gates(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    row = _normalize_epoch_v3(metadata)
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epoch_number": row.get("epoch_number"),
        "epoch_index": row.get("epoch_index"),
        "candidate_exists": row.get("candidate_exists"),
        "evaluation_mode": row.get("evaluation_mode"),
        "truth_state": row.get("truth_state"),
        "gate_rule_versions": row.get("gate_rule_versions", {}),
        "gates": row.get("gates", {}),
        "rejection_reason_codes": row.get("rejection_reason_codes", []),
    }


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}/mutations")
async def training_v3_run_epoch_mutations(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    row = _normalize_epoch_v3(metadata)
    decisions = list(metadata.get("prompt_scoring") or [])
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epoch_number": row.get("epoch_number"),
        "epoch_index": row.get("epoch_index"),
        "candidate_exists": row.get("candidate_exists"),
        "evaluation_mode": row.get("evaluation_mode"),
        "truth_state": row.get("truth_state"),
        "mutation_accept_threshold": metadata.get("mutation_accept_threshold"),
        "decisions": decisions,
        "count": len(decisions),
    }


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}/selection")
async def training_v3_run_epoch_selection(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    row = _normalize_epoch_v3(metadata)
    stats = metadata.get("selection_stats", {})
    diagnostics = metadata.get("selection_diagnostics", [])
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epoch_number": row.get("epoch_number"),
        "epoch_index": row.get("epoch_index"),
        "candidate_exists": row.get("candidate_exists"),
        "evaluation_mode": row.get("evaluation_mode"),
        "truth_state": row.get("truth_state"),
        "selection_stats": stats,
        "selection_diagnostics": diagnostics,
        "rca_selection_diagnostics": (
            stats.get("rca_selection_diagnostics")
            if isinstance(stats, dict)
            else {}
        ),
    }


@app.get("/training/v3/runs/{run_id}/epochs/{epoch_number}/rca")
async def training_v3_run_epoch_rca(run_id: str, epoch_number: int) -> dict[str, Any]:
    if not ENABLE_TRAINING_V3_API:
        return {"status": "error", "message": "training v3 API is disabled", "schema_version": "3.0"}
    records = _load_generation_records(TRAINING_GENERATIONS_PATH)
    grouped = _group_records_by_run(records)
    run_records = grouped.get(str(run_id))
    if not run_records:
        return {"status": "error", "message": f"Run not found: {run_id}", "schema_version": "3.0"}
    _, metadata = _find_epoch_metadata_by_number(run_records, epoch_number)
    if metadata is None:
        return {
            "status": "error",
            "message": f"Epoch not found for run {run_id}: {epoch_number}",
            "schema_version": "3.0",
        }
    row = _normalize_epoch_v3(metadata)
    high_level = metadata.get("rca_high_level_summary") or {}
    return {
        "status": "ok",
        "schema_version": "3.0",
        "run_id": str(run_id),
        "epoch_number": row.get("epoch_number"),
        "epoch_index": row.get("epoch_index"),
        "candidate_exists": row.get("candidate_exists"),
        "evaluation_mode": row.get("evaluation_mode"),
        "truth_state": row.get("truth_state"),
        "summary": high_level.get("summary"),
        "themes": high_level.get("themes", []),
        "model_used": high_level.get("model_used"),
        "selection_rule": high_level.get("selection_rule"),
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
                        "ENABLE_TRAINING_V3_API": ENABLE_TRAINING_V3_API,
                        "ENABLE_TRAINING_TRUTH_FLAGS": ENABLE_TRAINING_TRUTH_FLAGS,
                        "ENABLE_TRAINING_UI_V3": ENABLE_TRAINING_UI_V3,
                        "ENABLE_RUNS_UI_V3": ENABLE_RUNS_UI_V3,
                        "ENABLE_PROMPT_INDEX_V3": ENABLE_PROMPT_INDEX_V3,
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
