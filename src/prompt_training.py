from __future__ import annotations

import copy
import json
import logging
import random
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    import utility as utility_module
    from pipeline import Pipeline
    from pipeline_blocks import _get_all_prompted_blocks
    from utility import generate_text
except Exception:  # pragma: no cover
    from src import utility as utility_module
    from src.pipeline import Pipeline
    from src.pipeline_blocks import _get_all_prompted_blocks
    from src.utility import generate_text


logger = logging.getLogger("prompt_training")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream_handler)
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "prompt_training.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)
    except Exception:
        # Keep training functional even if file logging cannot be initialized.
        pass


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return f"{text[:max_chars]}... [truncated {remaining} chars]"


def _json_safe(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
    max_depth: int = 10,
    max_items: int = 120,
) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if _depth >= max_depth:
        return f"<max_depth:{type(value).__name__}>"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    if isinstance(value, BaseModel):
        return _json_safe(
            value.model_dump(),
            _seen=_seen,
            _depth=_depth + 1,
            max_depth=max_depth,
            max_items=max_items,
        )

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_ref:dict>"
        _seen.add(obj_id)
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["<truncated_items>"] = f"{len(value) - max_items} omitted"
                break
            out[str(key)] = _json_safe(
                item,
                _seen=_seen,
                _depth=_depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        _seen.discard(obj_id)
        return out

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_ref:list>"
        _seen.add(obj_id)
        out_list: list[Any] = []
        sequence = list(value)
        for idx, item in enumerate(sequence):
            if idx >= max_items:
                out_list.append(f"<truncated_items:{len(sequence) - max_items}>")
                break
            out_list.append(
                _json_safe(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
            )
        _seen.discard(obj_id)
        return out_list

    if hasattr(value, "__dict__"):
        obj_id = id(value)
        if obj_id in _seen:
            return f"<circular_ref:{type(value).__name__}>"
        _seen.add(obj_id)
        payload = {
            "__class__": type(value).__name__,
            "attributes": _json_safe(
                vars(value),
                _seen=_seen,
                _depth=_depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            ),
        }
        _seen.discard(obj_id)
        return payload

    return str(value)


def _safe_json_dumps(value: Any, max_chars: int | None = None) -> str:
    serialized = _json_safe(value)
    text = json.dumps(serialized, ensure_ascii=True, default=str)
    return _truncate_text(text, max_chars)


def _estimated_tokens_from_text(text: str) -> int:
    # Rough approximation for provider budgeting decisions.
    return max(1, int(len(text or "") / 4))


def _should_skip_oss_for_prompt(prompt: str, token_budget: int = 5500) -> bool:
    return _estimated_tokens_from_text(prompt) > token_budget


def _extract_explicit_json_keys(prompt_text: str) -> set[str]:
    return set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', prompt_text or ""))


def _validate_prompt_contract(
    block_id: str,
    prompt_text: str,
    block_details: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    schema = block_details.get("schema")
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        return issues

    prompt_lower = (prompt_text or "").lower()
    requires_json = len(schema.model_fields) > 0
    required_fields = set(schema.model_fields.keys())

    if requires_json and "json" not in prompt_lower:
        issues.append("missing_json_output_instruction")

    if requires_json and (
        "plain text" in prompt_lower
        or "no json" in prompt_lower
        or "no json or markup" in prompt_lower
        or "no json or markdown" in prompt_lower
    ):
        issues.append("conflicting_plain_text_instruction")

    explicit_keys = _extract_explicit_json_keys(prompt_text)
    if explicit_keys:
        if required_fields and required_fields.isdisjoint(explicit_keys):
            issues.append("explicit_json_keys_do_not_match_schema")
        if block_id == "self_critique_block":
            if "weaknesses" in explicit_keys and not {
                "given_item",
                "general_critique",
                "list_of_issues",
            }.issubset(explicit_keys):
                issues.append("self_critique_schema_drift")

    if block_id == "self_critique_block":
        required_tokens = ("given_item", "general_critique", "list_of_issues")
        if not all(token in prompt_lower for token in required_tokens):
            issues.append("self_critique_missing_required_keys")

        drift_markers = ('"weaknesses"', "`weaknesses`", "weaknesses:")
        if any(marker in prompt_lower for marker in drift_markers):
            issues.append("self_critique_schema_drift")

        string_list_markers = (
            "array of strings",
            "list of strings",
            "strings only",
            "each issue must be a string",
            "plain strings only",
        )
        if not any(marker in prompt_lower for marker in string_list_markers):
            issues.append("self_critique_issue_item_type_unspecified")

    if block_id == "long_response_synthesis_block":
        if "plain text" in prompt_lower and "json" in prompt_lower:
            issues.append("long_response_conflicting_output_contract")

    if block_id == "sub_plan_creation_block":
        required_tokens = ("sub_plan", "steps", "tool_uses")
        if not all(token in prompt_lower for token in required_tokens):
            issues.append("sub_plan_missing_required_keys")

        if explicit_keys and "sub_plan" in explicit_keys and not {"steps", "tool_uses"}.issubset(explicit_keys):
            issues.append("sub_plan_schema_drift")

    if block_id in {"primary_tool_router_block", "secondary_tool_router_block"}:
        required_tokens = ("routes", "id", "inputs")
        if not all(token in prompt_lower for token in required_tokens):
            issues.append("router_missing_required_keys")

        route_list_markers = (
            "array of routes",
            "list of routes",
            "\"routes\": [",
            "`routes`",
        )
        if not any(marker in prompt_lower for marker in route_list_markers):
            issues.append("router_route_list_shape_unspecified")

        if explicit_keys and "id" in explicit_keys and "routes" not in explicit_keys:
            issues.append("router_single_route_schema_drift")

    return issues


def _prune_for_rca(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 8,
    max_items: int = 40,
    max_text_chars: int = 900,
) -> Any:
    if depth >= max_depth:
        return f"<max_depth:{type(value).__name__}>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_text(value, max_text_chars)

    if isinstance(value, BaseModel):
        return _prune_for_rca(
            value.model_dump(),
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            max_text_chars=max_text_chars,
        )

    if isinstance(value, list):
        out: list[Any] = []
        for idx, item in enumerate(value):
            if idx >= max_items:
                out.append(f"<truncated_items:{len(value) - max_items}>")
                break
            out.append(
                _prune_for_rca(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_text_chars=max_text_chars,
                )
            )
        return out

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        skip_keys = {"events", "event_count", "rca_bundle"}
        for idx, (raw_key, raw_item) in enumerate(value.items()):
            if idx >= max_items:
                out["<truncated_items>"] = f"{len(value) - max_items} omitted"
                break

            key = str(raw_key)
            if key in skip_keys:
                continue

            item = raw_item

            if key == "data_uri":
                out[key] = "<omitted_data_uri>"
                continue

            if key == "image_embeddings" and isinstance(item, list):
                compact_images: list[dict[str, Any]] = []
                for image in item[:8]:
                    if not isinstance(image, dict):
                        continue
                    compact_images.append(
                        {
                            "index": image.get("index"),
                            "path": image.get("path"),
                            "media_type": image.get("media_type"),
                        }
                    )
                out[key] = compact_images
                continue

            if key == "generated_code" and isinstance(item, str):
                out["generated_code_preview"] = _truncate_text(item, 240)
                continue

            if key == "attempt_log" and isinstance(item, list):
                compact_attempts: list[dict[str, Any]] = []
                for attempt in item[:5]:
                    if not isinstance(attempt, dict):
                        continue
                    compact_attempts.append(
                        {
                            "attempt": attempt.get("attempt"),
                            "success": attempt.get("success"),
                            "selected_packages": attempt.get("selected_packages", []),
                            "error": _truncate_text(str(attempt.get("error", "")), 240),
                            "output_preview": _truncate_text(str(attempt.get("output_preview", "")), 240),
                            "generated_code_preview": _truncate_text(
                                str(attempt.get("generated_code", "")),
                                240,
                            ),
                        }
                    )
                out[key] = compact_attempts
                if len(item) > 5:
                    out["attempt_log_truncated"] = len(item) - 5
                continue

            out[key] = _prune_for_rca(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_text_chars=max_text_chars,
            )
        return out

    return str(value)


def _compact_run_output_for_storage(run_output: dict[str, Any]) -> dict[str, Any]:
    compact = _prune_for_rca(run_output, max_depth=8, max_items=40, max_text_chars=900)
    if isinstance(compact, dict):
        return compact
    return {"response": str(compact)}


def _score_stats(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    sorted_scores = sorted(float(item) for item in scores)
    count = len(sorted_scores)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    else:
        median = sorted_scores[mid]

    return {
        "count": count,
        "avg": sum(sorted_scores) / count,
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "median": median,
    }


class GenericGradeSchema(BaseModel):
    prompt_alignment_score: int = Field(ge=1, le=10)
    factuality_score: int = Field(ge=1, le=10)
    clarity_score: int = Field(ge=1, le=10)
    helpfulness_score: int = Field(ge=1, le=10)
    safety_score: int = Field(ge=1, le=10)
    tool_usage_score: int = Field(ge=1, le=10)
    major_issues: str
    strengths: str


class BlockAnalysisSchema(BaseModel):
    block_id: str
    generic_issue_score: int = Field(ge=1, le=10)
    criteria_misalignment_score: int = Field(ge=1, le=10)
    need_fix: bool
    analysis: str
    what_to_fix: str


class PipelineAnalysisReportSchema(BaseModel):
    overall_recommendations: str
    block_analyses: list[BlockAnalysisSchema]


class PromptMutationSchema(BaseModel):
    plan_for_improvement: str
    prompt: str


class PromptCriteriaScoreSchema(BaseModel):
    generic_quality_score: int = Field(ge=1, le=10)
    criteria_alignment_score: int = Field(ge=1, le=10)
    anti_overfit_score: int = Field(ge=1, le=10)
    notes: str


class GeneralizerCheckSchema(BaseModel):
    overfit_risk_score: int = Field(ge=1, le=10)
    suspicious_phrases: list[str]
    rationale: str


def _contains_degraded_marker(value: Any) -> bool:
    if isinstance(value, str):
        return "[Graceful degradation]" in value
    if isinstance(value, dict):
        return any(_contains_degraded_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_degraded_marker(item) for item in value)
    if hasattr(value, "model_dump"):
        return _contains_degraded_marker(value.model_dump())
    return False


def _generate_with_oss_fallback(
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.2,
) -> tuple[BaseModel, str]:
    attempts: list[tuple[str, BaseModel]] = []
    model_order: list[str] = ["oss120b", "nemotron"]
    if _should_skip_oss_for_prompt(prompt):
        logger.info("Skipping oss120b for oversized prompt; using nemotron directly")
        model_order = ["nemotron"]

    for model in model_order:
        retries = 2 if model == "oss120b" else 5
        max_retry_wait = 8.0 if model == "oss120b" else 30.0
        output = generate_text(
            prompt=prompt,
            model=model,
            schema=schema,
            temperature=temperature,
            retries=retries,
            max_total_retry_wait=max_retry_wait,
        )
        attempts.append((model, output))
        if not _contains_degraded_marker(output):
            return output, model

    fallback_model, fallback_output = attempts[-1]
    return fallback_output, fallback_model


@contextmanager
def prompt_override_context(prompt_suite: dict[str, str]):
    old_path = utility_module.LOADED_PROMPTS.get("path", "")
    old_content = copy.deepcopy(utility_module.LOADED_PROMPTS.get("content", {}))
    utility_module.LOADED_PROMPTS["path"] = "<in-memory-prompt-suite>"
    utility_module.LOADED_PROMPTS["content"] = dict(prompt_suite)
    try:
        yield
    finally:
        utility_module.LOADED_PROMPTS["path"] = old_path
        utility_module.LOADED_PROMPTS["content"] = old_content


def _placeholder_tokens(text: str) -> set[str]:
    simple = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", text))
    triple = set(re.findall(r"\{\{\{([a-zA-Z0-9_]+)\}\}\}", text))
    double = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", text))
    return simple.union(triple).union(double)


def get_prompted_block_details() -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for block_class in _get_all_prompted_blocks():
        block_details = copy.deepcopy(getattr(block_class, "details", {}))
        block_id = block_details.get("id")
        if block_id:
            details[block_id] = block_details
    return details


def load_test_cases(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Training dataset must be a list: {path}")
    return data


def load_prompts_file(path: str) -> dict[str, str]:
    return json.loads(Path(path).read_text())


def save_prompts_file(path: str, prompts: dict[str, str]) -> None:
    Path(path).write_text(json.dumps(prompts, indent=2))


class PromptSuiteStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"generations": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {"generations": []}

    def save_generation(self, prompts: dict[str, str], generation: int, metadata: dict[str, Any]) -> None:
        record = {
            "generation": generation,
            "prompts": prompts,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.data.setdefault("generations", []).append(record)
        self.path.write_text(json.dumps(self.data, indent=2))

    def list_generations(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for record in self.data.get("generations", []):
            metadata = record.get("metadata", {})
            summaries.append(
                {
                    "generation": record.get("generation"),
                    "timestamp": record.get("timestamp"),
                    "run_id": metadata.get("run_id"),
                    "avg_score_a": metadata.get("avg_score_a"),
                    "avg_score_b": metadata.get("avg_score_b"),
                    "winner": metadata.get("winner"),
                    "changed_keys": metadata.get("changed_keys", []),
                }
            )
        return summaries

    def get_generation(self, generation: int, run_id: str | None = None) -> dict[str, str] | None:
        for record in reversed(self.data.get("generations", [])):
            if int(record.get("generation", -1)) == int(generation):
                metadata = record.get("metadata", {})
                if run_id and str(metadata.get("run_id", "")) != str(run_id):
                    continue
                prompts = record.get("prompts")
                if isinstance(prompts, dict):
                    return prompts
        return None


class TrainingProgressTracker:
    def __init__(
        self,
        status_path: str,
        events_path: str,
        run_id: str,
        epochs_total: int,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.epochs_total = int(max(0, epochs_total))
        self.seq = 0
        self._run_started_perf = time.perf_counter()
        self._phase_started_perf = self._run_started_perf
        self._last_phase = "init"
        self.status_path = Path(status_path)
        self.events_path = Path(events_path)
        self.status: dict[str, Any] = {
            "run_id": run_id,
            "state": "idle",
            "started_at": _iso_now(),
            "updated_at": _iso_now(),
            "epoch_current": 0,
            "epochs_total": self.epochs_total,
            "phase": "init",
            "step": "init",
            "message": "",
            "events_count": 0,
            "last_event_type": None,
            "current_case": None,
            "elapsed_seconds": 0.0,
            "phase_elapsed_seconds": 0.0,
            "metrics": {},
            "paths": {
                "status_path": str(self.status_path),
                "events_path": str(self.events_path),
                "log_path": str(Path("logs/prompt_training.log")),
            },
        }

        if not self.enabled:
            return

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        # Start each run with a fresh event log for clean streaming.
        self.events_path.write_text("")
        self._write_status()

    def _write_status(self) -> None:
        if not self.enabled:
            return
        tmp_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.status, indent=2))
        tmp_path.replace(self.status_path)

    def _append_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")

    def emit(
        self,
        event_type: str,
        phase: str,
        step: str,
        message: str,
        *,
        epoch_current: int | None = None,
        payload: dict[str, Any] | None = None,
        current_case: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _iso_now()
        self.seq += 1
        if phase != self._last_phase:
            self._last_phase = phase
            self._phase_started_perf = time.perf_counter()
        elapsed_seconds = max(0.0, time.perf_counter() - self._run_started_perf)
        phase_elapsed_seconds = max(0.0, time.perf_counter() - self._phase_started_perf)
        event = {
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": timestamp,
            "event_type": event_type,
            "phase": phase,
            "step": step,
            "message": message,
            "epoch_current": int(epoch_current or self.status.get("epoch_current", 0)),
            "epochs_total": self.epochs_total,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "phase_elapsed_seconds": round(phase_elapsed_seconds, 3),
            "payload": payload or {},
            "current_case": current_case,
            "metrics": metrics or {},
        }

        self.status["updated_at"] = timestamp
        self.status["phase"] = phase
        self.status["step"] = step
        self.status["message"] = message
        self.status["events_count"] = self.seq
        self.status["last_event_type"] = event_type
        self.status["elapsed_seconds"] = round(elapsed_seconds, 3)
        self.status["phase_elapsed_seconds"] = round(phase_elapsed_seconds, 3)
        if epoch_current is not None:
            self.status["epoch_current"] = int(epoch_current)
        if current_case is not None:
            self.status["current_case"] = current_case
        elif not str(step).startswith("case_"):
            self.status["current_case"] = None
        if metrics:
            merged = dict(self.status.get("metrics", {}))
            merged.update(metrics)
            self.status["metrics"] = merged
        if event_type in {"run_started", "run_resumed"}:
            self.status["state"] = "running"
        elif event_type == "run_completed":
            self.status["state"] = "completed"
        elif event_type == "run_error":
            self.status["state"] = "error"

        self._write_status()
        self._append_event(event)
        logger.info(
            f"[training:{event_type}] epoch={event.get('epoch_current')}/{self.epochs_total} "
            f"phase={phase} step={step} msg={message}"
        )

    def complete(self, metrics: dict[str, Any] | None = None) -> None:
        self.emit(
            "run_completed",
            phase="complete",
            step="finished",
            message="Training completed successfully",
            epoch_current=self.epochs_total,
            metrics=metrics or {},
            current_case=None,
        )

    def fail(self, error_message: str, epoch_current: int | None = None) -> None:
        self.emit(
            "run_error",
            phase="error",
            step="failed",
            message=error_message,
            epoch_current=epoch_current,
            payload={"error": error_message},
            current_case=None,
        )


def _should_short_circuit_grading(run_output: dict[str, Any]) -> bool:
    if not isinstance(run_output, dict):
        return False

    degraded = bool(run_output.get("degraded_mode_active"))
    failure_signals = run_output.get("tool_failure_signals")
    has_failure_signals = isinstance(failure_signals, list) and len(failure_signals) > 0
    response_text = str(run_output.get("response", "")).strip()

    if not degraded and not has_failure_signals:
        return False
    if response_text and len(response_text) >= 180:
        return False
    return True


def _short_circuit_grade(run_output: dict[str, Any]) -> dict[str, Any]:
    notes = run_output.get("degraded_notes")
    notes_text = "; ".join(str(item) for item in notes) if isinstance(notes, list) else str(notes or "")
    major = "Pipeline execution degraded due to tool/runtime failure; skipped model grading for efficiency."
    if notes_text:
        major = f"{major} Notes: {notes_text[:500]}"
    return {
        "prompt_alignment_score": 1,
        "factuality_score": 1,
        "clarity_score": 2,
        "helpfulness_score": 1,
        "safety_score": 5,
        "tool_usage_score": 1,
        "major_issues": major,
        "strengths": "",
        "aggregate_score": (1 + 1 + 2 + 1 + 5 + 1) / 6.0,
        "grading_mode": "short_circuit_failure",
    }


def grade_result(
    prompt: str,
    run_output: dict[str, Any],
    validation: str | None,
) -> dict[str, Any]:
    final_response = str(run_output.get("response", ""))
    grading_prompt = (
        "You are grading an AI pipeline response. Score strictly from 1 to 10.\n\n"
        f"User Prompt:\n{prompt}\n\n"
        f"Final Response:\n{final_response}\n\n"
        f"Validation Criteria (ground truth when available):\n{validation or 'N/A'}\n\n"
        "Rubric:\n"
        "- prompt_alignment_score: instruction compliance and scope match\n"
        "- factuality_score: correctness and non-hallucination\n"
        "- clarity_score: structure and readability\n"
        "- helpfulness_score: usefulness and actionability\n"
        "- safety_score: policy/safety compliance\n"
        "- tool_usage_score: appropriate and grounded use of tool evidence\n"
        "Also provide major_issues and strengths."
    )

    graded = generate_text(
        prompt=grading_prompt,
        model="nemotron",
        schema=GenericGradeSchema,
        temperature=0.0,
        retries=5,
        max_total_retry_wait=30.0,
    )

    payload = graded.model_dump()
    payload["aggregate_score"] = (
        payload["prompt_alignment_score"]
        + payload["factuality_score"]
        + payload["clarity_score"]
        + payload["helpfulness_score"]
        + payload["safety_score"]
        + payload["tool_usage_score"]
    ) / 6.0
    return payload


def root_cause_analysis(
    prompt: str,
    run_output: dict[str, Any],
    major_issues: str,
    valid_blocks: list[str],
    block_details_map: dict[str, dict[str, Any]],
    criteria_context_text: str | None = None,
) -> list[BlockAnalysisSchema]:
    if criteria_context_text is None:
        criteria_sections: list[str] = []
        for block_id in valid_blocks:
            details = block_details_map.get(block_id, {})
            criteria = details.get("prompt_creation_parameters", {})
            criteria_sections.append(
                f"Block: {block_id}\nPrompt criteria:\n{_safe_json_dumps(criteria, max_chars=1200)}"
            )
        criteria_context_text = "\n\n".join(criteria_sections)

    compact_run_output = _compact_run_output_for_storage(run_output)
    run_output_json = _safe_json_dumps(compact_run_output, max_chars=22000)

    rca_prompt = (
        "You are an expert pipeline RCA analyst.\n"
        "Use BOTH generic quality failures and block-specific prompt criteria misalignment.\n"
        "You MUST only reference valid block IDs.\n\n"
        f"Valid blocks:\n{', '.join(valid_blocks)}\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Major issues from grading:\n{major_issues}\n\n"
        f"Full run output (for RCA):\n{run_output_json}\n\n"
        f"Block prompt criteria context:\n{criteria_context_text}\n"
    )

    report, used_model = _generate_with_oss_fallback(
        prompt=rca_prompt,
        schema=PipelineAnalysisReportSchema,
        temperature=0.3,
    )
    logger.info(f"RCA generated with model={used_model}")

    analyses: list[BlockAnalysisSchema] = []
    for item in report.block_analyses:
        if item.block_id in valid_blocks:
            analyses.append(item)
    return analyses


def score_prompt_against_criteria(
    block_id: str,
    prompt_text: str,
    block_details: dict[str, Any],
) -> tuple[PromptCriteriaScoreSchema, str]:
    criteria = block_details.get("prompt_creation_parameters", {})
    score_prompt = (
        "Score this pipeline-block prompt using both generic quality and criteria fit.\n"
        f"Block ID: {block_id}\n\n"
        f"Prompt text:\n{prompt_text}\n\n"
        f"Prompt creation criteria:\n{_safe_json_dumps(criteria, max_chars=2500)}\n\n"
        "Return numeric scores for generic_quality_score, criteria_alignment_score, anti_overfit_score, and notes."
    )
    result, used_model = _generate_with_oss_fallback(
        prompt=score_prompt,
        schema=PromptCriteriaScoreSchema,
        temperature=0.1,
    )
    return result, used_model


def prompt_improvements(
    current_prompts: dict[str, str],
    block_analyses: list[BlockAnalysisSchema],
    block_details_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    grouped: dict[str, list[BlockAnalysisSchema]] = defaultdict(list)
    for analysis in block_analyses:
        if analysis.need_fix:
            grouped[analysis.block_id].append(analysis)

    improvements: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], tuple[PromptCriteriaScoreSchema, str]] = {}

    def _score_cached(
        block_id: str,
        prompt_text: str,
        block_details: dict[str, Any],
    ) -> tuple[PromptCriteriaScoreSchema, str]:
        cache_key = (block_id, prompt_text)
        cached = score_cache.get(cache_key)
        if cached is not None:
            return cached
        scored = score_prompt_against_criteria(block_id, prompt_text, block_details)
        score_cache[cache_key] = scored
        return scored

    for block_id, analyses in grouped.items():
        if block_id not in current_prompts:
            continue

        current_prompt = current_prompts[block_id]
        details = block_details_map.get(block_id, {})
        criteria = details.get("prompt_creation_parameters", {})

        unique_analyses: list[BlockAnalysisSchema] = []
        seen_analysis_signatures: set[tuple[str, str]] = set()
        for analysis in analyses:
            signature = (
                str(analysis.analysis).strip().lower(),
                str(analysis.what_to_fix).strip().lower(),
            )
            if signature in seen_analysis_signatures:
                continue
            seen_analysis_signatures.add(signature)
            unique_analyses.append(analysis)

        required_placeholders = sorted(_placeholder_tokens(current_prompt))
        analyses_text = "\n".join(
            (
                f"- generic_issue_score={analysis.generic_issue_score}, "
                f"criteria_misalignment_score={analysis.criteria_misalignment_score}\n"
                f"  analysis={analysis.analysis}\n"
                f"  what_to_fix={analysis.what_to_fix}"
            )
            for analysis in unique_analyses
        )

        improve_prompt = (
            "You are an expert prompt engineer. Improve this block prompt.\n"
            "Use both generic quality and the block's prompt_creation_parameters.\n"
            "Do not insert specific training-case facts, numbers, or direct answer fragments.\n\n"
            f"Block ID: {block_id}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"RCA analyses:\n{analyses_text}\n\n"
            f"Prompt creation parameters:\n{_safe_json_dumps(criteria, max_chars=2500)}\n\n"
            f"Required placeholders to preserve: {required_placeholders}\n"
            "Return a revised prompt that remains generalizable."
        )

        mutation, used_model = _generate_with_oss_fallback(
            prompt=improve_prompt,
            schema=PromptMutationSchema,
            temperature=0.5,
        )
        logger.info(f"Prompt mutation for {block_id} generated with model={used_model}")

        candidate_prompt = mutation.prompt
        if candidate_prompt.strip() == current_prompt.strip():
            improvements[block_id] = current_prompt
            diagnostics.append(
                {
                    "block_id": block_id,
                    "analysis_count": len(unique_analyses),
                    "mutation_model": used_model,
                    "changed": False,
                    "accepted": False,
                    "decision_reason": "candidate_same_as_baseline",
                    "required_placeholders_count": len(required_placeholders),
                    "missing_placeholders": [],
                    "baseline_total": None,
                    "candidate_total": None,
                    "delta_total": 0.0,
                    "baseline_scores": {},
                    "candidate_scores": {},
                    "score_models": {},
                    "candidate_prompt_preview": candidate_prompt[:240],
                }
            )
            continue

        missing = [
            token for token in required_placeholders if token not in _placeholder_tokens(candidate_prompt)
        ]
        if missing:
            logger.warning(f"Prompt mutation for {block_id} missing placeholders {missing}; keeping baseline")
            improvements[block_id] = current_prompt
            diagnostics.append(
                {
                    "block_id": block_id,
                    "analysis_count": len(unique_analyses),
                    "mutation_model": used_model,
                    "changed": True,
                    "accepted": False,
                    "decision_reason": "missing_required_placeholders",
                    "required_placeholders_count": len(required_placeholders),
                    "missing_placeholders": missing,
                    "baseline_total": None,
                    "candidate_total": None,
                    "delta_total": 0.0,
                    "baseline_scores": {},
                    "candidate_scores": {},
                    "score_models": {},
                    "candidate_prompt_preview": candidate_prompt[:240],
                }
            )
            continue

        contract_issues = _validate_prompt_contract(block_id, candidate_prompt, details)
        if contract_issues:
            logger.warning(
                f"Prompt mutation for {block_id} violated prompt contract {contract_issues}; keeping baseline"
            )
            improvements[block_id] = current_prompt
            diagnostics.append(
                {
                    "block_id": block_id,
                    "analysis_count": len(unique_analyses),
                    "mutation_model": used_model,
                    "changed": True,
                    "accepted": False,
                    "decision_reason": "prompt_contract_violation",
                    "contract_issues": contract_issues,
                    "required_placeholders_count": len(required_placeholders),
                    "missing_placeholders": [],
                    "baseline_total": None,
                    "candidate_total": None,
                    "delta_total": 0.0,
                    "baseline_scores": {},
                    "candidate_scores": {},
                    "score_models": {},
                    "candidate_prompt_preview": candidate_prompt[:240],
                }
            )
            continue

        baseline_score, baseline_score_model = _score_cached(block_id, current_prompt, details)
        candidate_score, candidate_score_model = _score_cached(block_id, candidate_prompt, details)

        baseline_total = (
            baseline_score.generic_quality_score
            + baseline_score.criteria_alignment_score
            + baseline_score.anti_overfit_score
        )
        candidate_total = (
            candidate_score.generic_quality_score
            + candidate_score.criteria_alignment_score
            + candidate_score.anti_overfit_score
        )
        delta_total = candidate_total - baseline_total
        accepted = candidate_total >= baseline_total
        if accepted:
            improvements[block_id] = candidate_prompt
            decision_reason = "candidate_score_ge_baseline"
        else:
            improvements[block_id] = current_prompt
            decision_reason = "candidate_score_lt_baseline"

        diagnostics.append(
            {
                "block_id": block_id,
                "analysis_count": len(unique_analyses),
                "mutation_model": used_model,
                "changed": True,
                "accepted": accepted,
                "decision_reason": decision_reason,
                "required_placeholders_count": len(required_placeholders),
                "missing_placeholders": [],
                "baseline_total": baseline_total,
                "candidate_total": candidate_total,
                "delta_total": delta_total,
                "baseline_scores": baseline_score.model_dump(),
                "candidate_scores": candidate_score.model_dump(),
                "score_models": {
                    "baseline": baseline_score_model,
                    "candidate": candidate_score_model,
                },
                "candidate_prompt_preview": candidate_prompt[:240],
            }
        )

    return improvements, diagnostics


def _heuristic_leakage_scan(prompts: dict[str, str], selected_cases: list[dict[str, Any]]) -> list[str]:
    suspicious: list[str] = []
    haystack = "\n".join(prompts.values()).lower()

    for case in selected_cases:
        for field in ("prompt", "validation"):
            text = str(case.get(field, "")).strip().lower()
            if not text:
                continue
            chunks = [part.strip() for part in re.split(r"[\n\.;]", text) if len(part.strip()) > 24]
            for chunk in chunks:
                if chunk in haystack:
                    suspicious.append(chunk[:160])

    return sorted(set(suspicious))


def generalizer_check(prompts: dict[str, str], selected_cases: list[dict[str, Any]]) -> dict[str, Any]:
    suspicious = _heuristic_leakage_scan(prompts, selected_cases)

    check_prompt = (
        "Assess prompt-suite overfitting risk.\n"
        "Risk is high if prompts contain direct fragments of training prompts/validations or encode narrow solutions.\n\n"
        f"Prompts:\n{_safe_json_dumps(prompts, max_chars=20000)}\n\n"
        f"Sampled training cases:\n{_safe_json_dumps(selected_cases, max_chars=12000)}\n\n"
        f"Heuristic suspicious fragments:\n{suspicious}\n"
    )

    check, model_used = _generate_with_oss_fallback(
        prompt=check_prompt,
        schema=GeneralizerCheckSchema,
        temperature=0.1,
    )
    logger.info(f"Generalizer check generated with model={model_used}")

    merged_suspicious = sorted(set(check.suspicious_phrases + suspicious))
    return {
        "overfit_risk_score": max(
            int(check.overfit_risk_score),
            8 if merged_suspicious else int(check.overfit_risk_score),
        )
        if merged_suspicious
        else int(check.overfit_risk_score),
        "suspicious_phrases": merged_suspicious,
        "rationale": check.rationale,
    }


def evaluate_suite(
    prompt_suite: dict[str, str],
    selected_cases: list[dict[str, Any]],
    thinking_level: Literal["low", "med-synth", "med-plan", "high"],
    phase: str = "evaluation",
    epoch_current: int | None = None,
    progress_tracker: TrainingProgressTracker | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    results: list[dict[str, Any]] = []
    scores: list[float] = []

    total_cases = len(selected_cases)
    with prompt_override_context(prompt_suite):
        pipeline = Pipeline(
            enable_image_embedding=False,
            allow_visual_outputs=False,
        )

        for index, case in enumerate(selected_cases, start=1):
            case_prompt = str(case.get("prompt", ""))
            validation = case.get("validation")
            case_meta = {
                "index": index,
                "total": total_cases,
                "id": case.get("id"),
                "category": case.get("category"),
                "difficulty": case.get("difficulty"),
                "prompt_preview": case_prompt[:140],
            }

            if progress_tracker:
                progress_tracker.emit(
                    event_type="case_started",
                    phase=phase,
                    step="case_run",
                    message=f"Running case {index}/{total_cases}",
                    epoch_current=epoch_current,
                    current_case=case_meta,
                    payload={"validation_preview": str(validation or "")[:180]},
                )

            case_started_perf = time.perf_counter()
            try:
                run_started_perf = time.perf_counter()
                run_output = pipeline.run(
                    prompt=case_prompt,
                    thinking_level=thinking_level,
                    include_events=False,
                )
                run_elapsed_ms = (time.perf_counter() - run_started_perf) * 1000.0

                if _should_short_circuit_grading(run_output):
                    grade_started_perf = time.perf_counter()
                    grade = _short_circuit_grade(run_output)
                    grading_elapsed_ms = (time.perf_counter() - grade_started_perf) * 1000.0
                else:
                    grade_started_perf = time.perf_counter()
                    grade = grade_result(case_prompt, run_output, validation)
                    grading_elapsed_ms = (time.perf_counter() - grade_started_perf) * 1000.0

                case_elapsed_ms = (time.perf_counter() - case_started_perf) * 1000.0

                score = float(grade.get("aggregate_score", 0.0))
                scores.append(score)
                compact_run_output = _compact_run_output_for_storage(run_output)
                results.append(
                    {
                        "prompt": case_prompt,
                        "validation": validation,
                        "run_output": compact_run_output,
                        "grade": grade,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "prompt": case_prompt,
                        "validation": validation,
                        "run_output": {
                            "response": "",
                            "degraded_mode_active": True,
                            "degraded_notes": [str(exc)],
                        },
                        "grade": {
                            "aggregate_score": 0.0,
                            "major_issues": str(exc),
                            "strengths": "",
                        },
                    }
                )
                scores.append(0.0)
                if progress_tracker:
                    case_elapsed_ms = (time.perf_counter() - case_started_perf) * 1000.0
                    progress_tracker.emit(
                        event_type="case_completed",
                        phase=phase,
                        step="case_error",
                        message=f"Case {index}/{total_cases} failed",
                        epoch_current=epoch_current,
                        current_case=case_meta,
                        payload={
                            "error": str(exc),
                            "timing_ms": {
                                "case_total": round(case_elapsed_ms, 2),
                            },
                        },
                        metrics={"latest_case_score": 0.0},
                    )
                continue

            if progress_tracker:
                degraded = bool(run_output.get("degraded_mode_active"))
                score_breakdown = {
                    "prompt_alignment_score": grade.get("prompt_alignment_score"),
                    "factuality_score": grade.get("factuality_score"),
                    "clarity_score": grade.get("clarity_score"),
                    "helpfulness_score": grade.get("helpfulness_score"),
                    "safety_score": grade.get("safety_score"),
                    "tool_usage_score": grade.get("tool_usage_score"),
                }
                step_coverage = run_output.get("step_coverage", {})
                step_coverage_ratio = (
                    float(step_coverage.get("coverage_ratio"))
                    if isinstance(step_coverage, dict) and isinstance(step_coverage.get("coverage_ratio"), (int, float))
                    else None
                )
                failure_signals = run_output.get("tool_failure_signals")
                response_text = str(run_output.get("response", ""))
                citation_links = run_output.get("citation_links")
                image_paths = run_output.get("image_paths")
                image_embeddings = run_output.get("image_embeddings")
                progress_tracker.emit(
                    event_type="case_completed",
                    phase=phase,
                    step="case_scored",
                    message=f"Case {index}/{total_cases} scored",
                    epoch_current=epoch_current,
                    current_case=case_meta,
                    payload={
                        "degraded_mode_active": degraded,
                        "grading_mode": grade.get("grading_mode", "llm"),
                        "aggregate_score": score,
                        "score_breakdown": score_breakdown,
                        "major_issues_preview": str(grade.get("major_issues", ""))[:320],
                        "strengths_preview": str(grade.get("strengths", ""))[:320],
                        "timing_ms": {
                            "pipeline_run": round(run_elapsed_ms, 2),
                            "grading": round(grading_elapsed_ms, 2),
                            "case_total": round(case_elapsed_ms, 2),
                        },
                        "response_chars": len(response_text),
                        "tool_failure_signals_count": len(failure_signals) if isinstance(failure_signals, list) else 0,
                        "step_coverage_ratio": step_coverage_ratio,
                        "citation_count": len(citation_links) if isinstance(citation_links, dict) else 0,
                        "image_path_count": len(image_paths) if isinstance(image_paths, list) else 0,
                        "embedded_image_count": len(image_embeddings) if isinstance(image_embeddings, list) else 0,
                    },
                    metrics={"latest_case_score": scores[-1]},
                )

    return results, scores


def train_ab_loop(
    base_prompts_path: str,
    output_path: str,
    test_cases_path: str,
    epochs: int = 10,
    num_test_cases_per_trial: int = 5,
    random_seed: int = 42,
    thinking_level: Literal["low", "med-synth", "med-plan", "high"] = "med-synth",
    fail_threshold: float = 8.8,
    progress_status_path: str = "data/training_status.json",
    progress_events_path: str = "data/training_events.jsonl",
    track_progress: bool = True,
) -> dict[str, Any]:
    random.seed(random_seed)

    current_prompts = load_prompts_file(base_prompts_path)
    test_cases = load_test_cases(test_cases_path)
    block_details_map = get_prompted_block_details()

    store = PromptSuiteStore(output_path)
    run_id = str(uuid.uuid4())
    generation = 0
    store.save_generation(current_prompts, generation, {"note": "initial", "run_id": run_id})
    tracker = TrainingProgressTracker(
        status_path=progress_status_path,
        events_path=progress_events_path,
        run_id=run_id,
        epochs_total=epochs,
        enabled=track_progress,
    )

    logger.info(
        f"Starting prompt A/B training: epochs={epochs}, sampled_cases={num_test_cases_per_trial}, thinking={thinking_level}"
    )
    tracker.emit(
        event_type="run_started",
        phase="init",
        step="startup",
        message="Training run initialized",
        epoch_current=0,
        payload={
            "base_prompts_path": base_prompts_path,
            "output_path": output_path,
            "test_cases_path": test_cases_path,
            "epochs": epochs,
            "num_test_cases_per_trial": num_test_cases_per_trial,
            "thinking_level": thinking_level,
            "fail_threshold": fail_threshold,
            "random_seed": random_seed,
            "total_available_cases": len(test_cases),
        },
    )

    try:
        for epoch in range(epochs):
            epoch_current = epoch + 1
            tracker.emit(
                event_type="epoch_started",
                phase="epoch",
                step="start",
                message=f"Epoch {epoch_current}/{epochs} started",
                epoch_current=epoch_current,
                metrics={"generation": generation},
            )

            selected = random.sample(test_cases, k=min(num_test_cases_per_trial, len(test_cases)))
            tracker.emit(
                event_type="sample_selected",
                phase="sampling",
                step="select_cases",
                message=f"Selected {len(selected)} case(s) for epoch {epoch_current}",
                epoch_current=epoch_current,
                payload={
                    "sampled_case_ids": [case.get("id") for case in selected],
                    "sampled_categories": sorted({case.get("category") for case in selected if case.get("category")}),
                },
            )

            logger.info(f"Epoch {epoch_current}/{epochs} - Phase A baseline")
            tracker.emit(
                event_type="phase_started",
                phase="phase_a_eval",
                step="evaluate_baseline",
                message=f"Epoch {epoch_current}: evaluating baseline prompts",
                epoch_current=epoch_current,
            )
            results_a, scores_a = evaluate_suite(
                current_prompts,
                selected,
                thinking_level,
                phase="phase_a_eval",
                epoch_current=epoch_current,
                progress_tracker=tracker,
            )
            avg_score_a = sum(scores_a) / max(len(scores_a), 1)
            stats_a = _score_stats(scores_a)
            tracker.emit(
                event_type="phase_completed",
                phase="phase_a_eval",
                step="evaluate_baseline_done",
                message=f"Epoch {epoch_current}: baseline evaluation completed",
                epoch_current=epoch_current,
                payload={"score_stats": stats_a},
                metrics={
                    "avg_score_a": avg_score_a,
                    "phase_a_min_score": stats_a["min"],
                    "phase_a_max_score": stats_a["max"],
                    "phase_a_median_score": stats_a["median"],
                },
            )

            failed = [r for r in results_a if float(r["grade"].get("aggregate_score", 0.0)) < fail_threshold]
            all_analyses: list[BlockAnalysisSchema] = []
            valid_block_ids = sorted(list(current_prompts.keys()))
            criteria_sections = []
            for block_id in valid_block_ids:
                details = block_details_map.get(block_id, {})
                criteria = details.get("prompt_creation_parameters", {})
                criteria_sections.append(
                    f"Block: {block_id}\nPrompt criteria:\n{_safe_json_dumps(criteria, max_chars=1200)}"
                )
            criteria_context_text = "\n\n".join(criteria_sections)
            tracker.emit(
                event_type="rca_batch_started",
                phase="rca",
                step="identify_failures",
                message=f"Epoch {epoch_current}: running RCA on {len(failed)} failed case(s)",
                epoch_current=epoch_current,
                payload={
                    "failed_cases": len(failed),
                    "failed_case_prompts": [
                        str(item.get("prompt", ""))[:180]
                        for item in failed[:20]
                    ],
                },
            )

            for failed_index, failed_case in enumerate(failed, start=1):
                prompt_preview = str(failed_case.get("prompt", ""))[:140]
                tracker.emit(
                    event_type="rca_started",
                    phase="rca",
                    step="case_rca",
                    message=f"Epoch {epoch_current}: RCA {failed_index}/{len(failed)}",
                    epoch_current=epoch_current,
                    current_case={
                        "index": failed_index,
                        "total": len(failed),
                        "prompt_preview": prompt_preview,
                    },
                )
                analyses = root_cause_analysis(
                    prompt=failed_case["prompt"],
                    run_output=failed_case["run_output"],
                    major_issues=str(failed_case["grade"].get("major_issues", "")),
                    valid_blocks=valid_block_ids,
                    block_details_map=block_details_map,
                    criteria_context_text=criteria_context_text,
                )
                all_analyses.extend(analyses)
                tracker.emit(
                    event_type="rca_completed",
                    phase="rca",
                    step="case_rca_done",
                    message=f"Epoch {epoch_current}: RCA {failed_index}/{len(failed)} complete",
                    epoch_current=epoch_current,
                    payload={"analyses_added": len(analyses)},
                )

            candidate_prompts = dict(current_prompts)
            tracker.emit(
                event_type="mutation_started",
                phase="prompt_improvement",
                step="generate_mutations",
                message=f"Epoch {epoch_current}: generating prompt mutations",
                epoch_current=epoch_current,
                payload={"rca_items": len(all_analyses)},
            )
            improvements, prompt_scoring = prompt_improvements(
                current_prompts=current_prompts,
                block_analyses=all_analyses,
                block_details_map=block_details_map,
            )
            candidate_prompts.update(improvements)

            changed_keys = [
                key for key in candidate_prompts.keys() if candidate_prompts.get(key) != current_prompts.get(key)
            ]
            scored_blocks = [item for item in prompt_scoring if item.get("changed")]
            accepted_blocks = [item for item in scored_blocks if item.get("accepted")]
            rejected_blocks = [item for item in scored_blocks if not item.get("accepted")]
            baseline_totals = [
                float(item.get("baseline_total"))
                for item in scored_blocks
                if isinstance(item.get("baseline_total"), (int, float))
            ]
            candidate_totals = [
                float(item.get("candidate_total"))
                for item in scored_blocks
                if isinstance(item.get("candidate_total"), (int, float))
            ]
            delta_totals = [
                float(item.get("delta_total"))
                for item in scored_blocks
                if isinstance(item.get("delta_total"), (int, float))
            ]

            tracker.emit(
                event_type="prompt_scoring_snapshot",
                phase="prompt_improvement",
                step="score_candidates",
                message=f"Epoch {epoch_current}: prompt scoring diagnostics ready",
                epoch_current=epoch_current,
                payload={
                    "block_scores": prompt_scoring,
                    "accepted_block_ids": [item.get("block_id") for item in accepted_blocks],
                    "rejected_block_ids": [item.get("block_id") for item in rejected_blocks],
                },
                metrics={
                    "prompt_scored_blocks": len(scored_blocks),
                    "prompt_accepted_blocks": len(accepted_blocks),
                    "prompt_rejected_blocks": len(rejected_blocks),
                    "prompt_score_baseline_avg": (sum(baseline_totals) / len(baseline_totals)) if baseline_totals else 0.0,
                    "prompt_score_candidate_avg": (sum(candidate_totals) / len(candidate_totals)) if candidate_totals else 0.0,
                    "prompt_score_delta_avg": (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0,
                },
            )
            tracker.emit(
                event_type="mutation_completed",
                phase="prompt_improvement",
                step="mutations_ready",
                message=f"Epoch {epoch_current}: mutation pass complete",
                epoch_current=epoch_current,
                payload={
                    "changed_keys": changed_keys,
                    "prompt_scored_blocks": len(scored_blocks),
                    "prompt_accepted_blocks": len(accepted_blocks),
                    "prompt_rejected_blocks": len(rejected_blocks),
                },
            )

            generalizer = None
            if (epoch_current) % 5 == 0:
                tracker.emit(
                    event_type="generalizer_started",
                    phase="generalizer_check",
                    step="run_generalizer",
                    message=f"Epoch {epoch_current}: running generalizer check",
                    epoch_current=epoch_current,
                )
                generalizer = generalizer_check(candidate_prompts, selected)
                tracker.emit(
                    event_type="generalizer_completed",
                    phase="generalizer_check",
                    step="run_generalizer_done",
                    message=f"Epoch {epoch_current}: generalizer check complete",
                    epoch_current=epoch_current,
                    payload=generalizer,
                )
                if int(generalizer.get("overfit_risk_score", 0)) >= 8:
                    logger.warning("Generalizer check flagged overfit risk; rejecting candidate mutations this epoch")
                    candidate_prompts = dict(current_prompts)
                    changed_keys = []
                    tracker.emit(
                        event_type="generalizer_rejected_mutations",
                        phase="generalizer_check",
                        step="reject_mutations",
                        message=f"Epoch {epoch_current}: mutations rejected by generalizer",
                        epoch_current=epoch_current,
                    )

            if changed_keys:
                logger.info(f"Epoch {epoch_current}/{epochs} - Phase B candidate")
                tracker.emit(
                    event_type="phase_started",
                    phase="phase_b_eval",
                    step="evaluate_candidate",
                    message=f"Epoch {epoch_current}: evaluating candidate prompts",
                    epoch_current=epoch_current,
                    payload={"changed_keys": changed_keys},
                )
                results_b, scores_b = evaluate_suite(
                    candidate_prompts,
                    selected,
                    thinking_level,
                    phase="phase_b_eval",
                    epoch_current=epoch_current,
                    progress_tracker=tracker,
                )
                avg_score_b = sum(scores_b) / max(len(scores_b), 1)
                stats_b = _score_stats(scores_b)
                tracker.emit(
                    event_type="phase_completed",
                    phase="phase_b_eval",
                    step="evaluate_candidate_done",
                    message=f"Epoch {epoch_current}: candidate evaluation completed",
                    epoch_current=epoch_current,
                    payload={"score_stats": stats_b},
                    metrics={
                        "avg_score_b": avg_score_b,
                        "phase_b_min_score": stats_b["min"],
                        "phase_b_max_score": stats_b["max"],
                        "phase_b_median_score": stats_b["median"],
                    },
                )
            else:
                results_b, scores_b = results_a, scores_a
                avg_score_b = avg_score_a
                tracker.emit(
                    event_type="phase_skipped",
                    phase="phase_b_eval",
                    step="evaluate_candidate_skipped",
                    message=f"Epoch {epoch_current}: candidate evaluation skipped (no changes)",
                    epoch_current=epoch_current,
                )

            winner = "baseline"
            improvement_delta = avg_score_b - avg_score_a
            prompt_delta_avg = (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0
            if changed_keys and avg_score_b > avg_score_a and prompt_delta_avg >= 0.0:
                winner = "candidate"
                generation += 1
                current_prompts = candidate_prompts

            metadata = {
                "epoch": epoch,
                "run_id": run_id,
                "avg_score_a": avg_score_a,
                "avg_score_b": avg_score_b,
                "improvement_delta": improvement_delta,
                "winner": winner,
                "changed_keys": changed_keys if winner == "candidate" else [],
                "scores_a": scores_a,
                "scores_b": scores_b,
                "thinking_level": thinking_level,
                "num_failures_in_a": len(failed),
                "num_rca_items": len(all_analyses),
                "generalizer": generalizer,
                "prompt_scoring": prompt_scoring,
                "prompt_scoring_summary": {
                    "scored_blocks": len(scored_blocks),
                    "accepted_blocks": len(accepted_blocks),
                    "rejected_blocks": len(rejected_blocks),
                    "baseline_avg_total": (sum(baseline_totals) / len(baseline_totals)) if baseline_totals else 0.0,
                    "candidate_avg_total": (sum(candidate_totals) / len(candidate_totals)) if candidate_totals else 0.0,
                    "delta_avg_total": (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0,
                },
            }
            store.save_generation(current_prompts, generation, metadata)
            tracker.emit(
                event_type="epoch_completed",
                phase="epoch",
                step="end",
                message=f"Epoch {epoch_current}/{epochs} completed",
                epoch_current=epoch_current,
                payload={
                    "winner": winner,
                    "changed_keys": metadata["changed_keys"],
                    "num_failures_in_a": len(failed),
                    "num_rca_items": len(all_analyses),
                },
                metrics={
                    "avg_score_a": avg_score_a,
                    "avg_score_b": avg_score_b,
                    "improvement_delta": improvement_delta,
                    "generation": generation,
                    "num_failures_in_a": len(failed),
                    "num_rca_items": len(all_analyses),
                },
            )

        save_prompts_file(base_prompts_path, current_prompts)
        result = {
            "status": "ok",
            "run_id": run_id,
            "generation": generation,
            "output_path": output_path,
            "base_prompts_path": base_prompts_path,
            "history_entries": len(PromptSuiteStore(output_path).list_generations()),
            "progress_status_path": progress_status_path,
            "progress_events_path": progress_events_path,
        }
        tracker.complete(metrics={"generation": generation, "history_entries": result["history_entries"]})
        return result
    except Exception as exc:
        tracker.fail(str(exc))
        logger.exception(f"Training run failed: {exc}")
        raise


def run_training_loop(
    base_prompts_path: str = "data/prompts.json",
    output_path: str = "data/prompt_suite_generations.json",
    test_cases_path: str = "data/prompt_train_cases.json",
    epochs: int = 10,
    num_test_cases_per_trial: int = 5,
    random_seed: int = 42,
    thinking_level: Literal["low", "med-synth", "med-plan", "high"] = "med-synth",
    progress_status_path: str = "data/training_status.json",
    progress_events_path: str = "data/training_events.jsonl",
    track_progress: bool = True,
) -> dict[str, Any]:
    return train_ab_loop(
        base_prompts_path=base_prompts_path,
        output_path=output_path,
        test_cases_path=test_cases_path,
        epochs=epochs,
        num_test_cases_per_trial=num_test_cases_per_trial,
        random_seed=random_seed,
        thinking_level=thinking_level,
        progress_status_path=progress_status_path,
        progress_events_path=progress_events_path,
        track_progress=track_progress,
    )


if __name__ == "__main__":
    result = run_training_loop()
    print(json.dumps(result, indent=2))
