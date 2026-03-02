from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    import utility as utility_module
    from pipeline import Pipeline
    from pipeline_blocks import _get_all_prompted_blocks
    from pipeline_blocks import get_tool_prompt_descriptors
    from utility import generate_text
except Exception:  # pragma: no cover
    from src import utility as utility_module
    from src.pipeline import Pipeline
    from src.pipeline_blocks import _get_all_prompted_blocks
    from src.pipeline_blocks import get_tool_prompt_descriptors
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


ENABLE_CANDIDATE_GATES = _env_flag("ENABLE_CANDIDATE_GATES", default=True)
ENABLE_TOOL_PROMPT_TEMPLATES = _env_flag("ENABLE_TOOL_PROMPT_TEMPLATES", default=True)
ENABLE_MUTATION_RETRY_V2 = _env_flag("ENABLE_MUTATION_RETRY_V2", default=True)
ENABLE_TAIL_RISK_GATE = _env_flag("ENABLE_TAIL_RISK_GATE", default=True)
ENABLE_NO_CHANGE_STRICT_SEMANTICS = _env_flag("ENABLE_NO_CHANGE_STRICT_SEMANTICS", default=True)
ENABLE_BLOCK_IMPACT_RANKER = _env_flag("ENABLE_BLOCK_IMPACT_RANKER", default=True)
ENABLE_RESOURCE_TELEMETRY_V3 = _env_flag("ENABLE_RESOURCE_TELEMETRY_V3", default=True)
ENABLE_ADAPTIVE_EVAL_V1 = _env_flag("ENABLE_ADAPTIVE_EVAL_V1", default=True)
ENABLE_BALANCED_GUARDRAILS_V1 = _env_flag("ENABLE_BALANCED_GUARDRAILS_V1", default=True)
ENABLE_MUTATION_BUDGET_OPT_V1 = _env_flag("ENABLE_MUTATION_BUDGET_OPT_V1", default=True)
ENABLE_REPLAY_REBALANCE_V1 = _env_flag("ENABLE_REPLAY_REBALANCE_V1", default=True)
TRAINING_CASE_TIME_BUDGET_SECONDS = _env_float("TRAINING_CASE_TIME_BUDGET_SECONDS", 600.0)
STALL_WARNING_THRESHOLD_SECONDS = _env_float("TRAINING_STALL_WARNING_SECONDS", 600.0)

RUNTIME_GATE_MEAN_MAX_RATIO = 1.60
RUNTIME_GATE_P90_MAX_RATIO = 1.90
RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS = 180.0
STABILITY_GATE_MAX_TOOL_FAILURE_DELTA = 0.90
DEGRADATION_GATE_MAX_DELTA = 0.35
PROMPT_SCORE_ACCEPTANCE_MIN_DELTA = 2.0
MUTATION_ACCEPTANCE_TOTAL_MIN = 24
QUALITY_GATE_MIN_MEAN_DELTA = 0.0
QUALITY_GATE_MIN_CI_LOWER = -2.5
QUALITY_GATE_MIN_WIN_RATE = 0.50
QUALITY_GATE_MIN_P10_DELTA = -1.25
QUALITY_GATE_MIN_WORST_CASE_DELTA = -3.0
TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD = -3.0
TAIL_RISK_MAX_CATASTROPHIC_REGRESSIONS = 1
HOLDOUT_GATE_MIN_MEAN_DELTA = 0.0
HOLDOUT_GATE_MIN_CI_LOWER = -0.10
HOLDOUT_GATE_MIN_WIN_RATE = 0.50
EARLY_STOP_MIN_CASES = 6
EARLY_STOP_MEAN_DELTA_THRESHOLD = -0.75
DEFAULT_HOLDOUT_SPLIT_RATIO = 0.2
DEFAULT_MUTATION_MAX_RETRIES = 3
DEFAULT_MUTATION_TOURNAMENT_SIZE = 3
DEFAULT_MUTATION_DIVERSITY_TARGET = 3
MUTATION_RETRY_TEMPERATURE_SCHEDULE = [0.45, 0.30, 0.20, 0.10]
GENERALIZER_RISK_REJECT_THRESHOLD = 8
GENERALIZER_SUSPICIOUS_DELTA_THRESHOLD = 5
DEFAULT_RCA_THRESHOLD = 7.0
DEFAULT_RCA_FALLBACK_FRACTION = 0.5
DEFAULT_SELECTION_MODE = "hybrid_coverage_replay"
EVENT_SCHEMA_VERSION = "2.0"
METADATA_SCHEMA_VERSION = "2.0"
DEFAULT_DATASET_SCHEMA_VERSION = "2.0"
DEFAULT_REPLAY_FRACTION = 0.3
DEFAULT_EXPLORATION_FRACTION = 0.2
DEFAULT_REPLAY_COOLDOWN_EPOCHS = 2
DEFAULT_EVAL_MODE = "adaptive_sequential"
DEFAULT_TRAIN_EVAL_MIN_PAIRS = 4
DEFAULT_TRAIN_EVAL_MAX_PAIRS = 10
DEFAULT_TRAIN_EVAL_CHECKPOINTS = [4, 6, 8, 10]
DEFAULT_HOLDOUT_EVAL_MIN_PAIRS = 4
DEFAULT_HOLDOUT_EVAL_MAX_PAIRS = 8
DEFAULT_HOLDOUT_EVAL_CHECKPOINTS = [4, 6, 8]
DEFAULT_MUTATION_STAGE_A_TOP_K = 4
DEFAULT_MUTATION_PRECHECK_CASES = 4
DEFAULT_RESOURCE_TARGET_REDUCTION = 0.35

REPLAY_CAUSE_WEIGHTS: dict[str, float] = {
    "low_score": 1.8,
    "negative_delta": 2.0,
    "degraded": 1.2,
    "timeout": 2.5,
}

_TIME_LIMIT_MODE_WARNINGS: set[str] = set()


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


def _parse_eval_checkpoints(
    checkpoints: list[int] | tuple[int, ...] | str | None,
    *,
    default: list[int],
) -> list[int]:
    values: list[int] = []
    if isinstance(checkpoints, str):
        for raw in checkpoints.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                values.append(int(raw))
            except Exception:
                continue
    elif isinstance(checkpoints, (list, tuple)):
        for item in checkpoints:
            try:
                values.append(int(item))
            except Exception:
                continue
    if not values:
        values = list(default)
    return sorted({max(1, int(value)) for value in values})


class RunResourceTelemetry:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        self.phase_wall_seconds: dict[str, float] = {}
        self._phase_stack: list[tuple[str, float]] = []

    @contextmanager
    def phase(self, phase_name: str):
        phase = str(phase_name or "unknown")
        started = time.perf_counter()
        self._phase_stack.append((phase, started))
        try:
            yield
        finally:
            ended = time.perf_counter()
            if self._phase_stack:
                self._phase_stack.pop()
            elapsed = max(0.0, ended - started)
            self.phase_wall_seconds[phase] = self.phase_wall_seconds.get(phase, 0.0) + elapsed

    def current_phase(self) -> str:
        if self._phase_stack:
            return self._phase_stack[-1][0]
        return "unknown"

    def record_llm_call(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        self.llm_calls += 1
        self.llm_tokens_in += max(0, int(tokens_in))
        self.llm_tokens_out += max(0, int(tokens_out))

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": int(self.llm_calls),
            "llm_tokens_in": int(self.llm_tokens_in),
            "llm_tokens_out": int(self.llm_tokens_out),
            "phase_wall_seconds": {
                phase: round(float(seconds), 6)
                for phase, seconds in sorted(self.phase_wall_seconds.items())
            },
        }

    def delta_since(self, baseline: dict[str, Any]) -> dict[str, Any]:
        baseline_calls = int((baseline or {}).get("llm_calls") or 0)
        baseline_tokens_in = int((baseline or {}).get("llm_tokens_in") or 0)
        baseline_tokens_out = int((baseline or {}).get("llm_tokens_out") or 0)
        baseline_phase_wall = (baseline or {}).get("phase_wall_seconds") or {}
        delta_phase_wall: dict[str, float] = {}
        if isinstance(baseline_phase_wall, dict):
            for phase in set(self.phase_wall_seconds.keys()).union(
                {str(item) for item in baseline_phase_wall.keys()}
            ):
                current = float(self.phase_wall_seconds.get(phase, 0.0) or 0.0)
                previous = float(baseline_phase_wall.get(phase, 0.0) or 0.0)
                delta = current - previous
                if delta:
                    delta_phase_wall[str(phase)] = round(delta, 6)
        return {
            "llm_calls": int(self.llm_calls - baseline_calls),
            "llm_tokens_in": int(self.llm_tokens_in - baseline_tokens_in),
            "llm_tokens_out": int(self.llm_tokens_out - baseline_tokens_out),
            "phase_wall_seconds": dict(sorted(delta_phase_wall.items())),
        }


_ACTIVE_RESOURCE_TELEMETRY: RunResourceTelemetry | None = None


def _should_skip_oss_for_prompt(prompt: str, token_budget: int = 6000) -> bool:
    """Skip oss120b if estimated prompt tokens exceed budget (8k model limit)."""
    return _estimated_tokens_from_text(prompt) > token_budget


def _extract_explicit_json_keys(prompt_text: str) -> set[str]:
    """Extract JSON keys only from fenced code blocks and inline JSON object literals."""
    text = prompt_text or ""
    keys: set[str] = set()
    # Fenced code blocks (```json ... ``` or ``` ... ```)
    for block in re.findall(r'```[\w]*\s*\n?(.*?)```', text, re.DOTALL):
        keys.update(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', block))
    # Inline JSON object literals ({...})
    for block in re.findall(r'\{[^{}]*\}', text):
        keys.update(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', block))
    return keys


HARD_CONTRACT_ISSUES: set[str] = {
    "missing_json_output_instruction",
    "conflicting_plain_text_instruction",
    "explicit_json_keys_do_not_match_schema",
    "self_critique_missing_required_keys",
    "self_critique_schema_drift",
    "sub_plan_missing_required_keys",
    "sub_plan_schema_drift",
    "router_missing_required_keys",
    "router_route_list_shape_unspecified",
    "router_single_route_schema_drift",
    "long_response_conflicting_output_contract",
}


def _normalize_contract_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("\u2011", "-")
    normalized = re.sub(r"[^a-z0-9_\-\[\]`]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _contains_marker_groups(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    return all(_contains_any_marker(text, group) for group in groups)


def _split_contract_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    hard = [issue for issue in issues if issue in HARD_CONTRACT_ISSUES]
    soft = [issue for issue in issues if issue not in HARD_CONTRACT_ISSUES]
    return hard, soft


def _auto_repair_contract_issues(
    block_id: str,
    prompt_text: str,
    soft_issues: list[str],
) -> tuple[str, list[str]]:
    if not soft_issues:
        return prompt_text, []

    append_lines: list[str] = []
    applied: list[str] = []
    soft_set = set(soft_issues)

    if "tool_prompt_missing_contract_sections" in soft_set:
        append_lines.extend(
            [
                "Objective: restate the task objective exactly.",
                "Hard constraints: obey schema, avoid unsupported claims, and keep output bounded.",
                "Output contract: return strict JSON matching the schema keys.",
                "Uncertainty behavior: when evidence is weak or missing, state uncertainty directly.",
            ]
        )
        applied.append("tool_prompt_missing_contract_sections")

    if block_id == "web_search_tool_block" and "web_search_prompt_missing_evidence_constraints" in soft_set:
        append_lines.extend(
            [
                "Evidence constraint: cite only supplied reference IDs and do not invent references.",
                "If evidence is insufficient, state limitations plainly instead of extrapolating.",
            ]
        )
        applied.append("web_search_prompt_missing_evidence_constraints")

    if block_id == "wikipedia_search_tool_block" and "wikipedia_prompt_missing_insufficient_relevance_fallback" in soft_set:
        append_lines.append(
            'If relevance is weak, explicitly include the phrase "insufficient relevance" and provide nearest relevant context.'
        )
        applied.append("wikipedia_prompt_missing_insufficient_relevance_fallback")

    if block_id == "creative_idea_generator_tool_block" and "creative_prompt_missing_feasibility_tagging" in soft_set:
        append_lines.append("Each idea must include a feasibility tag: [feasibility: high|medium|low].")
        applied.append("creative_prompt_missing_feasibility_tagging")

    if block_id.startswith("deductive_reasoning_") and "deductive_prompt_missing_validation_dependency" in soft_set:
        append_lines.append("Use only validated premises for downstream deductions and validity checks.")
        applied.append("deductive_prompt_missing_validation_dependency")

    if block_id == "python_code_execution_tool_block" and "python_tool_prompt_missing_repair_or_schema_markers" in soft_set:
        append_lines.extend([
            'Output contract: return strict JSON with keys "code_to_run" and "packages_needed".',
            "Disallow file system writes, network access, and shell commands.",
            "If a previous error occurred, fix the root cause instead of masking it.",
        ])
        applied.append("python_tool_prompt_missing_repair_or_schema_markers")

    if not append_lines:
        return prompt_text, []

    repaired = prompt_text.rstrip() + "\n\nContract Repair Addendum:\n- " + "\n- ".join(append_lines)
    return repaired, applied


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
    prompt_norm = _normalize_contract_text(prompt_text or "")
    requires_json = len(schema.model_fields) > 0
    required_fields = set(schema.model_fields.keys())

    if requires_json and not _contains_any_marker(prompt_norm, ("json", "strict json", "valid json")):
        issues.append("missing_json_output_instruction")

    if requires_json and (
        "plain text" in prompt_norm
        or "no json" in prompt_norm
        or "no json or markup" in prompt_norm
        or "no json or markdown" in prompt_norm
    ):
        _benign_plain_text_contexts = (
            "explain in plain text", "describe in plain text",
            "plain text reasoning", "plain text before",
            "include plain text", "plain text explanation",
            "plain text summary", "reasoning in plain text",
        )
        has_json_instruction = _contains_any_marker(prompt_norm, ("json", "strict json", "valid json"))
        has_benign_context = any(ctx in prompt_norm for ctx in _benign_plain_text_contexts)
        if not (has_benign_context and has_json_instruction):
            issues.append("conflicting_plain_text_instruction")

    explicit_keys = _extract_explicit_json_keys(prompt_text)
    if explicit_keys:
        if len(explicit_keys) >= 3 and required_fields and required_fields.isdisjoint(explicit_keys):
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
        if not any(marker in prompt_norm for marker in string_list_markers):
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

        route_list_groups = (
            ("routes", "`routes`", "\"routes\""),
            ("array", "list", "[]"),
        )
        if not _contains_marker_groups(prompt_norm, route_list_groups):
            issues.append("router_route_list_shape_unspecified")

        if explicit_keys and "id" in explicit_keys and "routes" not in explicit_keys:
            issues.append("router_single_route_schema_drift")

    tool_prompt_ids = {
        "web_search_tool_block",
        "wikipedia_search_tool_block",
        "python_code_execution_tool_block",
        "creative_idea_generator_tool_block",
        "deductive_reasoning_premise_tool_block",
        "deductive_reasoning_confirm_premise_tool_block",
        "deductive_reasoning_conclusion_tool_block",
        "deductive_reasoning_conclusion_confirmation_tool_block",
    }
    if block_id in tool_prompt_ids:
        contract_groups = (
            ("objective", "goal"),
            ("hard constraints", "constraints", "non negotiable constraints"),
            ("output contract", "schema", "return strict json"),
            ("uncertainty behavior", "if uncertain", "state uncertainty"),
        )
        if not _contains_marker_groups(prompt_norm, contract_groups):
            issues.append("tool_prompt_missing_contract_sections")

    if block_id == "python_code_execution_tool_block":
        required_groups = (
            ("code_to_run",),
            ("packages_needed",),
            ("disallow", "forbid", "ban"),
            ("previous error", "retry repair", "previous_code", "fix root cause"),
        )
        if not _contains_marker_groups(prompt_norm, required_groups):
            issues.append("python_tool_prompt_missing_repair_or_schema_markers")

    if block_id == "web_search_tool_block":
        required_groups = (
            ("reference id", "reference ids", "citation"),
            ("do not invent", "do not fabricate", "no fabricated", "use only information present"),
        )
        if not _contains_marker_groups(prompt_norm, required_groups):
            issues.append("web_search_prompt_missing_evidence_constraints")

    if block_id == "wikipedia_search_tool_block":
        if not _contains_any_marker(prompt_norm, ("insufficient relevance", "weak relevance", "loosely relevant")):
            issues.append("wikipedia_prompt_missing_insufficient_relevance_fallback")

    if block_id == "creative_idea_generator_tool_block":
        if not _contains_any_marker(prompt_norm, ("feasibility", "feasible", "high medium low")):
            issues.append("creative_prompt_missing_feasibility_tagging")

    deductive_prompt_ids = {
        "deductive_reasoning_premise_tool_block",
        "deductive_reasoning_confirm_premise_tool_block",
        "deductive_reasoning_conclusion_tool_block",
        "deductive_reasoning_conclusion_confirmation_tool_block",
    }
    if block_id in deductive_prompt_ids and not _contains_any_marker(
        prompt_norm,
        ("validated premises", "premise validation", "valid premises"),
    ):
        issues.append("deductive_prompt_missing_validation_dependency")

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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(item) for item in values)
    index = max(0, min(len(sorted_values) - 1, int(round((percentile / 100.0) * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def _clamp_score(value: float, min_score: float = 1.0, max_score: float = 10.0) -> float:
    return float(max(min_score, min(max_score, value)))


def _case_identifier(case: dict[str, Any]) -> str:
    case_id = case.get("id")
    if case_id is not None and str(case_id).strip():
        return str(case_id).strip()
    prompt = str(case.get("prompt", "")).strip()
    validation = str(case.get("validation", "")).strip()
    return f"anon::{prompt[:180]}::{validation[:80]}"


def _case_bucket_key(case: dict[str, Any]) -> str:
    category = str(case.get("category") or "uncategorized").strip().lower()
    difficulty = str(case.get("difficulty") or "unspecified").strip().lower()
    return f"{category}::{difficulty}"


def build_case_buckets(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if not isinstance(case, dict):
            continue
        buckets[_case_bucket_key(case)].append(case)
    return dict(sorted(buckets.items(), key=lambda item: item[0]))


def stratified_split_cases(
    cases: list[dict[str, Any]],
    holdout_ratio: float = DEFAULT_HOLDOUT_SPLIT_RATIO,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = rng or random.Random()
    bounded_ratio = max(0.0, min(0.5, float(holdout_ratio)))
    buckets = build_case_buckets(cases)

    train_pool: list[dict[str, Any]] = []
    holdout_pool: list[dict[str, Any]] = []

    for bucket_cases in buckets.values():
        items = list(bucket_cases)
        rng.shuffle(items)
        if len(items) <= 1:
            train_pool.extend(items)
            continue

        holdout_count = int(round(len(items) * bounded_ratio))
        if holdout_count <= 0 and len(items) >= 4 and bounded_ratio > 0:
            holdout_count = 1
        holdout_count = max(0, min(holdout_count, len(items) - 1))

        holdout_pool.extend(items[:holdout_count])
        train_pool.extend(items[holdout_count:])

    if not train_pool and holdout_pool:
        train_pool.append(holdout_pool.pop())

    rng.shuffle(train_pool)
    rng.shuffle(holdout_pool)
    return train_pool, holdout_pool


def _build_sampling_state(cases: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    buckets = build_case_buckets(cases)
    shuffled_buckets: dict[str, list[dict[str, Any]]] = {}
    for bucket_key, bucket_cases in buckets.items():
        items = list(bucket_cases)
        rng.shuffle(items)
        shuffled_buckets[bucket_key] = items
    return {
        "buckets": shuffled_buckets,
        "bucket_order": list(shuffled_buckets.keys()),
        "bucket_offsets": {bucket_key: 0 for bucket_key in shuffled_buckets},
        "seen_case_ids": set(),
        "epoch_index": 0,
        "rng": rng,  # stored for re-shuffling on wrap-around
    }


def _next_case_from_bucket(
    *,
    state: dict[str, Any],
    bucket_key: str,
    selected_ids: set[str],
    prefer_unseen: bool,
) -> dict[str, Any] | None:
    buckets = state.get("buckets", {})
    if not isinstance(buckets, dict):
        return None
    bucket = buckets.get(bucket_key)
    if not isinstance(bucket, list) or not bucket:
        return None

    seen_case_ids = state.get("seen_case_ids", set())
    if not isinstance(seen_case_ids, set):
        seen_case_ids = set()
        state["seen_case_ids"] = seen_case_ids

    offsets = state.get("bucket_offsets", {})
    if not isinstance(offsets, dict):
        offsets = {}
        state["bucket_offsets"] = offsets

    start_index = int(offsets.get(bucket_key, 0) or 0) % len(bucket)
    # Re-shuffle when wrapping around to avoid repeating the same order
    if start_index == 0 and state.get("epoch_index", 0) > 0:
        shuffle_rng = state.get("rng")
        if shuffle_rng is not None:
            shuffle_rng.shuffle(bucket)
    for step in range(len(bucket)):
        idx = (start_index + step) % len(bucket)
        candidate = bucket[idx]
        if not isinstance(candidate, dict):
            continue
        case_id = _case_identifier(candidate)
        if case_id in selected_ids:
            continue
        is_unseen = case_id not in seen_case_ids
        if prefer_unseen and not is_unseen:
            continue
        offsets[bucket_key] = (idx + 1) % len(bucket)
        return candidate
    return None


def sample_cases_stratified(
    *,
    sample_state: dict[str, Any],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    target_size = max(0, int(sample_size))
    buckets = sample_state.get("buckets", {})
    if not isinstance(buckets, dict) or not buckets or target_size == 0:
        return []

    total_available = sum(len(items) for items in buckets.values() if isinstance(items, list))
    if total_available <= 0:
        return []

    target_size = min(target_size, total_available)

    bucket_order = sample_state.get("bucket_order", [])
    if not isinstance(bucket_order, list) or not bucket_order:
        bucket_order = sorted(str(key) for key in buckets.keys())
        sample_state["bucket_order"] = bucket_order

    epoch_index = int(sample_state.get("epoch_index", 0) or 0)
    rotation = epoch_index % len(bucket_order)
    rotated_order = bucket_order[rotation:] + bucket_order[:rotation]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for prefer_unseen in (True, False):
        made_progress = True
        while len(selected) < target_size and made_progress:
            made_progress = False
            for bucket_key in rotated_order:
                if len(selected) >= target_size:
                    break
                picked = _next_case_from_bucket(
                    state=sample_state,
                    bucket_key=bucket_key,
                    selected_ids=selected_ids,
                    prefer_unseen=prefer_unseen,
                )
                if not isinstance(picked, dict):
                    continue
                picked_id = _case_identifier(picked)
                selected.append(picked)
                selected_ids.add(picked_id)
                seen_case_ids = sample_state.setdefault("seen_case_ids", set())
                if isinstance(seen_case_ids, set):
                    seen_case_ids.add(picked_id)
                made_progress = True

    if len(selected) < target_size:
        fallback_pool: list[dict[str, Any]] = []
        for items in buckets.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if _case_identifier(item) in selected_ids:
                    continue
                fallback_pool.append(item)
        rng.shuffle(fallback_pool)
        for item in fallback_pool:
            if len(selected) >= target_size:
                break
            selected.append(item)
            selected_ids.add(_case_identifier(item))

    sample_state["epoch_index"] = epoch_index + 1
    return selected


def _all_cases_from_sampling_state(sample_state: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = sample_state.get("buckets", {})
    if not isinstance(buckets, dict):
        return []
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for bucket_cases in buckets.values():
        if not isinstance(bucket_cases, list):
            continue
        for case in bucket_cases:
            if not isinstance(case, dict):
                continue
            case_id = _case_identifier(case)
            if case_id in seen_ids:
                continue
            seen_ids.add(case_id)
            items.append(case)
    return items


def _replay_priority_score(history: dict[str, Any]) -> float:
    last_score = float(history.get("last_score") or 0.0)
    last_delta = float(history.get("last_delta") or 0.0)
    degraded_count = int(history.get("degraded_count") or 0)
    timeout_count = int(history.get("timeout_count") or 0)
    low_score_penalty = max(0.0, 7.0 - last_score)
    negative_delta_penalty = abs(last_delta) if last_delta < 0 else 0.0
    base_score = (
        (2.0 * low_score_penalty)
        + (2.5 * negative_delta_penalty)
        + (2.2 * degraded_count)
        + (3.5 * timeout_count)
    )
    replay_cause = "low_score"
    if timeout_count > 0:
        replay_cause = "timeout"
    elif negative_delta_penalty > 0:
        replay_cause = "negative_delta"
    elif degraded_count > 0:
        replay_cause = "degraded"
    cause_weight = float(REPLAY_CAUSE_WEIGHTS.get(replay_cause, 1.0))
    return float(base_score * cause_weight)


def select_cases_hybrid(
    *,
    sample_state: dict[str, Any],
    sample_size: int,
    rng: random.Random,
    case_history: dict[str, dict[str, Any]],
    epoch_current: int,
    replay_fraction: float = DEFAULT_REPLAY_FRACTION,
    exploration_fraction: float = DEFAULT_EXPLORATION_FRACTION,
    replay_cooldown_epochs: int = DEFAULT_REPLAY_COOLDOWN_EPOCHS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_size = max(0, int(sample_size))
    if target_size <= 0:
        return [], []

    replay_slots = max(0, int(round(target_size * max(0.0, min(0.6, replay_fraction)))))
    exploration_slots = max(0, int(round(target_size * max(0.0, min(0.6, exploration_fraction)))))
    coverage_slots = max(0, target_size - replay_slots - exploration_slots)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    diagnostics: list[dict[str, Any]] = []

    def _append(
        case: dict[str, Any],
        slot: str,
        reason: str,
        replay_cause: str | None = None,
    ) -> bool:
        case_id = _case_identifier(case)
        if case_id in selected_ids:
            return False
        selected.append(case)
        selected_ids.add(case_id)
        seen_case_ids = sample_state.setdefault("seen_case_ids", set())
        if isinstance(seen_case_ids, set):
            seen_case_ids.add(case_id)
        diagnostics.append(
            {
                "case_id": case_id,
                "slot": slot,
                "reason": reason,
                "replay_cause": replay_cause,
                "category": case.get("category"),
                "difficulty": case.get("difficulty"),
            }
        )
        return True

    coverage_cases = sample_cases_stratified(
        sample_state=sample_state,
        sample_size=coverage_slots,
        rng=rng,
    )
    for case in coverage_cases:
        _append(case, "coverage", "stratified_unseen_rotation")

    all_cases = _all_cases_from_sampling_state(sample_state)
    by_id = {_case_identifier(case): case for case in all_cases}

    replay_candidates: list[tuple[float, str, str]] = []
    for case_id, history in case_history.items():
        if case_id in selected_ids:
            continue
        if case_id not in by_id:
            continue
        last_epoch = int(history.get("last_epoch") or -999999)
        if epoch_current - last_epoch <= int(max(0, replay_cooldown_epochs)):
            continue
        if (
            float(history.get("last_score") or 10.0) >= 7.0
            and float(history.get("last_delta") or 0.0) >= 0.0
            and int(history.get("degraded_count") or 0) == 0
            and int(history.get("timeout_count") or 0) == 0
        ):
            continue
        replay_reason = "low_score"
        if int(history.get("timeout_count") or 0) > 0:
            replay_reason = "timeout"
        elif float(history.get("last_delta") or 0.0) < 0:
            replay_reason = "negative_delta"
        elif int(history.get("degraded_count") or 0) > 0:
            replay_reason = "degraded"
        replay_candidates.append((_replay_priority_score(history), case_id, replay_reason))
    replay_candidates.sort(key=lambda item: item[0], reverse=True)

    replay_selected_ids: set[str] = set()
    replay_count_by_cause: dict[str, int] = {}
    by_cause: dict[str, list[tuple[float, str, str]]] = {}
    for item in replay_candidates:
        by_cause.setdefault(item[2], []).append(item)

    mandatory_causes = ["negative_delta", "low_score"]
    if ENABLE_REPLAY_REBALANCE_V1:
        for cause in mandatory_causes:
            if len(replay_selected_ids) >= replay_slots:
                break
            candidate_list = by_cause.get(cause) or []
            if not candidate_list:
                continue
            _, case_id, replay_reason = candidate_list.pop(0)
            case = by_id.get(case_id)
            if not isinstance(case, dict):
                continue
            if _append(case, "replay", "replay_priority", replay_reason):
                replay_selected_ids.add(case_id)
                replay_count_by_cause[replay_reason] = replay_count_by_cause.get(replay_reason, 0) + 1

    cause_cap = max(1, int(math.ceil(max(1, replay_slots) * 0.6)))
    for _, case_id, replay_reason in replay_candidates:
        if len(replay_selected_ids) >= replay_slots:
            break
        if case_id in replay_selected_ids:
            continue
        if ENABLE_REPLAY_REBALANCE_V1 and replay_count_by_cause.get(replay_reason, 0) >= cause_cap:
            continue
        case = by_id.get(case_id)
        if not isinstance(case, dict):
            continue
        if _append(case, "replay", "replay_priority", replay_reason):
            replay_selected_ids.add(case_id)
            replay_count_by_cause[replay_reason] = replay_count_by_cause.get(replay_reason, 0) + 1

    if exploration_slots > 0:
        exploration_pool = [
            case for case in all_cases if _case_identifier(case) not in selected_ids
        ]
        weighted_pool: list[tuple[float, dict[str, Any]]] = []
        for case in exploration_pool:
            case_id = _case_identifier(case)
            sampled_times = int((case_history.get(case_id) or {}).get("times_sampled") or 0)
            novelty_weight = 1.0 / float(1 + sampled_times)
            weighted_pool.append((novelty_weight + rng.random() * 0.05, case))
        weighted_pool.sort(key=lambda item: item[0], reverse=True)
        for _, case in weighted_pool[:exploration_slots]:
            _append(case, "exploration", "novelty_weighted")

    if len(selected) < target_size:
        filler = sample_cases_stratified(
            sample_state=sample_state,
            sample_size=target_size,
            rng=rng,
        )
        for case in filler:
            if len(selected) >= target_size:
                break
            _append(case, "fallback", "stratified_fill")

    return selected[:target_size], diagnostics


def update_case_history(
    *,
    case_history: dict[str, dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    epoch_current: int,
) -> None:
    pair_count = min(len(selected_cases), len(results_a), len(results_b))
    for idx in range(pair_count):
        case = selected_cases[idx]
        case_id = _case_identifier(case)
        grade_a = results_a[idx].get("grade", {}) if isinstance(results_a[idx], dict) else {}
        grade_b = results_b[idx].get("grade", {}) if isinstance(results_b[idx], dict) else {}
        stats_a = results_a[idx].get("case_stats", {}) if isinstance(results_a[idx], dict) else {}
        stats_b = results_b[idx].get("case_stats", {}) if isinstance(results_b[idx], dict) else {}
        score_a = float(grade_a.get("aggregate_score", 0.0) or 0.0)
        score_b = float(grade_b.get("aggregate_score", 0.0) or 0.0)
        delta = score_b - score_a
        degraded = bool(stats_a.get("degraded_mode_active")) or bool(stats_b.get("degraded_mode_active"))
        timed_out = bool(stats_a.get("timed_out")) or bool(stats_b.get("timed_out"))

        entry = case_history.setdefault(case_id, {})
        entry["times_sampled"] = int(entry.get("times_sampled") or 0) + 1
        entry["last_epoch"] = int(epoch_current)
        entry["last_score"] = float(score_b)
        entry["last_delta"] = float(delta)
        entry["degraded_count"] = int(entry.get("degraded_count") or 0) + (1 if degraded else 0)
        entry["timeout_count"] = int(entry.get("timeout_count") or 0) + (1 if timed_out else 0)

def _paired_delta_stats(
    baseline_scores: list[float],
    candidate_scores: list[float],
    *,
    bootstrap_resamples: int = 1000,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    pair_count = min(len(baseline_scores), len(candidate_scores))
    if pair_count <= 0:
        return {
            "pair_count": 0,
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "deltas": [],
        }

    deltas = [float(candidate_scores[i]) - float(baseline_scores[i]) for i in range(pair_count)]
    mean_delta = float(sum(deltas) / pair_count)

    if pair_count == 1 or int(bootstrap_resamples) <= 1:
        return {
            "pair_count": pair_count,
            "mean_delta": mean_delta,
            "ci_lower": mean_delta,
            "ci_upper": mean_delta,
            "deltas": deltas,
        }

    local_rng = rng or random.Random(0)
    sample_means: list[float] = []
    for _ in range(int(bootstrap_resamples)):
        sample = [deltas[local_rng.randrange(pair_count)] for _ in range(pair_count)]
        sample_means.append(float(sum(sample) / pair_count))

    return {
        "pair_count": pair_count,
        "mean_delta": mean_delta,
        "ci_lower": _percentile(sample_means, 2.5) or mean_delta,
        "ci_upper": _percentile(sample_means, 97.5) or mean_delta,
        "deltas": deltas,
    }


def _should_early_stop_candidate_eval(
    *,
    baseline_scores: list[float],
    candidate_scores: list[float],
    min_cases: int = EARLY_STOP_MIN_CASES,
    mean_delta_threshold: float = EARLY_STOP_MEAN_DELTA_THRESHOLD,
) -> tuple[bool, dict[str, Any]]:
    pair_count = min(len(baseline_scores), len(candidate_scores))
    if pair_count <= 0:
        return False, {"pair_count": 0, "running_mean_delta": 0.0}

    deltas = [float(candidate_scores[i]) - float(baseline_scores[i]) for i in range(pair_count)]
    running_mean_delta = float(sum(deltas) / pair_count)
    triggered = pair_count >= max(1, int(min_cases)) and running_mean_delta < float(mean_delta_threshold)
    return triggered, {
        "pair_count": pair_count,
        "running_mean_delta": running_mean_delta,
        "threshold": float(mean_delta_threshold),
    }


def _timeout_enforcement_mode(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "disabled_no_budget"
    if not hasattr(signal, "SIGALRM"):
        return "disabled_no_sigalrm"
    if threading.current_thread() is not threading.main_thread():
        return "disabled_non_main_thread"
    return "signal_alarm"


@contextmanager
def _time_limit(seconds: float | None):
    mode = _timeout_enforcement_mode(seconds)
    if mode != "signal_alarm":
        if mode not in {"disabled_no_budget"} and mode not in _TIME_LIMIT_MODE_WARNINGS:
            logger.warning(f"Time limit guard inactive: mode={mode}")
            _TIME_LIMIT_MODE_WARNINGS.add(mode)
        yield
        return

    timer_seconds = float(seconds)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):  # type: ignore[unused-argument]
        raise TimeoutError(f"Case time budget exceeded ({timer_seconds:.1f}s)")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timer_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _iter_tool_execution_results(tool_context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(tool_context, dict):
        return []

    entries: list[dict[str, Any]] = []

    def _append_bundle(bundle: Any) -> None:
        if not isinstance(bundle, dict):
            return
        for item in bundle.get("results", []):
            if isinstance(item, dict):
                entries.append(item)

    _append_bundle(tool_context.get("primary_execution"))
    _append_bundle(tool_context.get("primary_continuity_execution"))

    for subplan in tool_context.get("subplans", []):
        if not isinstance(subplan, dict):
            continue
        _append_bundle(subplan.get("tool_execution"))
        _append_bundle(subplan.get("continuity_execution"))

    return entries


def _extract_case_tool_metrics(run_output: dict[str, Any]) -> dict[str, Any]:
    tool_context = run_output.get("tool_context") if isinstance(run_output, dict) else {}
    entries = _iter_tool_execution_results(tool_context if isinstance(tool_context, dict) else {})

    tool_invocations: dict[str, int] = {}
    tool_errors: dict[str, int] = {}
    python_attempts = 0
    python_failures = 0

    for entry in entries:
        resolved_id = str(entry.get("resolved_id") or entry.get("requested_id") or "unknown")
        tool_invocations[resolved_id] = tool_invocations.get(resolved_id, 0) + 1
        if entry.get("error"):
            tool_errors[resolved_id] = tool_errors.get(resolved_id, 0) + 1

        if resolved_id != "python_code_execution_tool_block":
            continue

        output = entry.get("output")
        if not isinstance(output, dict):
            continue
        diagnostics = output.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue

        attempts_run = diagnostics.get("attempts_run")
        attempts_allowed = diagnostics.get("attempts_allowed")
        final_success = bool(diagnostics.get("final_success"))
        final_error = diagnostics.get("final_error")

        if isinstance(attempts_run, int):
            python_attempts += max(0, attempts_run)
        elif isinstance(attempts_allowed, int):
            python_attempts += max(0, attempts_allowed)

        if not final_success:
            if isinstance(attempts_run, int) and attempts_run > 0:
                python_failures += attempts_run
            elif final_error:
                python_failures += 1

    signals = run_output.get("tool_failure_signals") if isinstance(run_output, dict) else []
    tool_failure_signals_count = len(signals) if isinstance(signals, list) else 0

    return {
        "tool_invocations": tool_invocations,
        "tool_errors": tool_errors,
        "python_exec_attempts": python_attempts,
        "python_exec_failures": python_failures,
        "tool_failure_signals_count": tool_failure_signals_count,
        "degraded_mode_active": bool(run_output.get("degraded_mode_active")) if isinstance(run_output, dict) else False,
    }


def _summarize_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    case_times_s: list[float] = []
    pipeline_times_s: list[float] = []
    grading_times_s: list[float] = []
    tool_failure_counts: list[float] = []
    degraded_flags: list[bool] = []
    python_attempts: list[float] = []
    python_failures: list[float] = []
    tool_invocation_totals: dict[str, int] = {}
    tool_error_totals: dict[str, int] = {}

    for result in results:
        case_stats = result.get("case_stats", {})
        if not isinstance(case_stats, dict):
            continue

        case_total_seconds = case_stats.get("case_total_seconds")
        if isinstance(case_total_seconds, (int, float)):
            case_times_s.append(float(case_total_seconds))

        pipeline_seconds = case_stats.get("pipeline_run_seconds")
        if isinstance(pipeline_seconds, (int, float)):
            pipeline_times_s.append(float(pipeline_seconds))

        grading_seconds = case_stats.get("grading_seconds")
        if isinstance(grading_seconds, (int, float)):
            grading_times_s.append(float(grading_seconds))

        failure_count = case_stats.get("tool_failure_signals_count")
        if isinstance(failure_count, (int, float)):
            tool_failure_counts.append(float(failure_count))

        degraded_flags.append(bool(case_stats.get("degraded_mode_active")))

        attempts = case_stats.get("python_exec_attempts")
        if isinstance(attempts, (int, float)):
            python_attempts.append(float(attempts))

        failures = case_stats.get("python_exec_failures")
        if isinstance(failures, (int, float)):
            python_failures.append(float(failures))

        invocations = case_stats.get("tool_invocations")
        if isinstance(invocations, dict):
            for tool_id, count in invocations.items():
                if not isinstance(count, (int, float)):
                    continue
                tool_invocation_totals[str(tool_id)] = tool_invocation_totals.get(str(tool_id), 0) + int(count)

        errors = case_stats.get("tool_errors")
        if isinstance(errors, dict):
            for tool_id, count in errors.items():
                if not isinstance(count, (int, float)):
                    continue
                tool_error_totals[str(tool_id)] = tool_error_totals.get(str(tool_id), 0) + int(count)

    case_count = len(results)
    degraded_count = sum(1 for item in degraded_flags if item)
    degraded_rate = (degraded_count / case_count) if case_count else 0.0

    return {
        "case_count": case_count,
        "mean_case_time_s": _mean(case_times_s) or 0.0,
        "p90_case_time_s": _percentile(case_times_s, 90) or 0.0,
        "sum_case_time_s": float(sum(case_times_s)),
        "sum_pipeline_run_s": float(sum(pipeline_times_s)),
        "sum_grading_s": float(sum(grading_times_s)),
        "avg_tool_failure_signals": _mean(tool_failure_counts) or 0.0,
        "degraded_case_rate": degraded_rate,
        "degraded_case_count": degraded_count,
        "avg_python_exec_attempts": _mean(python_attempts) or 0.0,
        "avg_python_exec_failures": _mean(python_failures) or 0.0,
        "tool_invocation_totals": tool_invocation_totals,
        "tool_error_totals": tool_error_totals,
    }


def _extract_case_id_from_result(result: dict[str, Any]) -> str:
    case_payload = result.get("case")
    if isinstance(case_payload, dict):
        return _case_identifier(case_payload)
    return str(result.get("prompt", ""))[:160]


def _paired_eval_summaries(
    phase_a_results: list[dict[str, Any]],
    phase_b_results: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    pair_count = min(len(phase_a_results), len(phase_b_results))
    if pair_count <= 0:
        return [], _summarize_eval_results([]), _summarize_eval_results([])

    paired_ids: list[str] = []
    paired_a: list[dict[str, Any]] = []
    paired_b: list[dict[str, Any]] = []
    for idx in range(pair_count):
        a_item = phase_a_results[idx]
        b_item = phase_b_results[idx]
        paired_ids.append(_extract_case_id_from_result(b_item))
        paired_a.append(a_item)
        paired_b.append(b_item)
    return paired_ids, _summarize_eval_results(paired_a), _summarize_eval_results(paired_b)


def _timeout_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(results)
    timeout_count = 0
    mode_counts: dict[str, int] = {}
    for item in results:
        case_stats = item.get("case_stats", {})
        if not isinstance(case_stats, dict):
            continue
        timeout_mode = str(case_stats.get("timeout_enforcement_mode") or "unknown")
        mode_counts[timeout_mode] = mode_counts.get(timeout_mode, 0) + 1
        if bool(case_stats.get("timed_out")):
            timeout_count += 1
    return {
        "timeout_count": timeout_count,
        "total_cases": total_cases,
        "timeout_rate": (float(timeout_count) / total_cases) if total_cases else 0.0,
        "timeout_enforcement_mode_counts": mode_counts,
    }


def _quality_gate_thresholds(n: int) -> tuple[float, float]:
    """Return (min_mean_delta, min_ci_lower) based on sample size tier."""
    if n >= 15:
        return (0.05, -0.50)
    if n >= 8:
        return (0.0, -1.0)
    return (0.0, -2.5)


def _stability_gate_thresholds(n: int) -> tuple[float, float]:
    """Return (max_tool_failure_delta, max_degraded_rate_delta) based on sample size tier."""
    if n >= 15:
        return (0.20, 0.05)
    if n >= 8:
        return (0.30, 0.10)
    return (0.50, 0.20)


def _gate_tier_label(n: int) -> str:
    if n >= 15:
        return "large"
    if n >= 8:
        return "medium"
    return "small"


def _evaluate_candidate_gates(
    *,
    delta_stats: dict[str, Any],
    phase_a_summary: dict[str, Any],
    phase_b_summary: dict[str, Any],
    enabled: bool,
    n_pairs: int = 0,
    quality_gate_min_mean_delta: float = QUALITY_GATE_MIN_MEAN_DELTA,
    quality_gate_min_win_rate: float = QUALITY_GATE_MIN_WIN_RATE,
    quality_gate_min_p10_delta: float = QUALITY_GATE_MIN_P10_DELTA,
    quality_gate_min_worst_case_delta: float = QUALITY_GATE_MIN_WORST_CASE_DELTA,
    runtime_gate_mean_ratio_max: float = RUNTIME_GATE_MEAN_MAX_RATIO,
    runtime_gate_p90_ratio_max: float = RUNTIME_GATE_P90_MAX_RATIO,
    runtime_gate_abs_mean_increase_max_seconds: float = RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS,
    stability_gate_tool_failure_delta_max: float = STABILITY_GATE_MAX_TOOL_FAILURE_DELTA,
    stability_gate_degraded_rate_delta_max: float = DEGRADATION_GATE_MAX_DELTA,
    tail_risk_gate_enabled: bool = ENABLE_TAIL_RISK_GATE,
    tail_risk_catastrophic_delta_threshold: float = TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD,
    tail_risk_max_catastrophic_regressions: int = TAIL_RISK_MAX_CATASTROPHIC_REGRESSIONS,
) -> dict[str, Any]:
    mean_delta = float(delta_stats.get("mean_delta", 0.0) or 0.0)
    ci_lower = float(delta_stats.get("ci_lower", mean_delta) or mean_delta)
    ci_upper = float(delta_stats.get("ci_upper", mean_delta) or mean_delta)

    deltas = [float(item) for item in (delta_stats.get("deltas") or []) if isinstance(item, (int, float))]
    wins = sum(1 for d in deltas if d > 0)
    win_rate = (wins / len(deltas)) if deltas else 0.0
    p10_delta = float(_percentile(deltas, 10) or mean_delta)
    worst_case_delta = float(min(deltas)) if deltas else mean_delta
    catastrophic_threshold = float(tail_risk_catastrophic_delta_threshold)
    catastrophic_regression_count = sum(1 for d in deltas if d <= catastrophic_threshold)

    tier = "balanced_lenient_v2"
    min_mean_delta = float(quality_gate_min_mean_delta)
    min_ci_lower = QUALITY_GATE_MIN_CI_LOWER
    min_win_rate = float(quality_gate_min_win_rate)
    min_p10_delta = float(quality_gate_min_p10_delta)
    min_worst_case_delta = float(quality_gate_min_worst_case_delta)
    quality_checks = {
        "mean_delta_nonnegative": mean_delta >= min_mean_delta,
        "win_rate_minimum": win_rate >= min_win_rate,
        "p10_delta_guardrail": p10_delta >= min_p10_delta,
        "worst_case_guardrail": worst_case_delta >= min_worst_case_delta,
    }
    quality_passed = all(bool(value) for value in quality_checks.values())
    quality_failed_checks = [name for name, passed in quality_checks.items() if not bool(passed)]

    quality_gate = {
        "name": "quality_gate",
        "passed": quality_passed,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "win_rate": round(win_rate, 4),
        "p10_delta": round(p10_delta, 6),
        "worst_case_delta": round(worst_case_delta, 6),
        "wins": wins,
        "n_pairs": n_pairs,
        "tier": tier,
        "thresholds": {
            "min_mean_delta": min_mean_delta,
            "min_ci_lower": min_ci_lower,
            "min_win_rate": min_win_rate,
            "min_p10_delta": min_p10_delta,
            "min_worst_case_delta": min_worst_case_delta,
        },
        "distance_from_threshold": {
            "mean_delta": round(mean_delta - min_mean_delta, 6),
            "ci_lower": round(ci_lower - min_ci_lower, 6),
            "win_rate": round(win_rate - min_win_rate, 4),
            "p10_delta": round(p10_delta - min_p10_delta, 6),
            "worst_case_delta": round(worst_case_delta - min_worst_case_delta, 6),
        },
        "checks": quality_checks,
        "failed_checks": quality_failed_checks,
        "passed_via": "all_quality_constraints" if quality_passed else "failed_quality_constraints",
        "rule": (
            f"mean_delta >= {min_mean_delta} and "
            f"win_rate >= {min_win_rate} and "
            f"p10_delta >= {min_p10_delta} and "
            f"worst_case_delta >= {min_worst_case_delta}"
        ),
    }

    mean_a = float(phase_a_summary.get("mean_case_time_s", 0.0) or 0.0)
    mean_b = float(phase_b_summary.get("mean_case_time_s", 0.0) or 0.0)
    p90_a = float(phase_a_summary.get("p90_case_time_s", 0.0) or 0.0)
    p90_b = float(phase_b_summary.get("p90_case_time_s", 0.0) or 0.0)
    failure_a = float(phase_a_summary.get("avg_tool_failure_signals", 0.0) or 0.0)
    failure_b = float(phase_b_summary.get("avg_tool_failure_signals", 0.0) or 0.0)
    degraded_rate_a = float(phase_a_summary.get("degraded_case_rate", 0.0) or 0.0)
    degraded_rate_b = float(phase_b_summary.get("degraded_case_rate", 0.0) or 0.0)

    mean_ratio = _safe_ratio(mean_b, mean_a)
    p90_ratio = _safe_ratio(p90_b, p90_a)
    mean_delta_seconds = mean_b - mean_a

    runtime_gate = {
        "name": "runtime_gate",
        "passed": (mean_ratio is None or mean_ratio <= float(runtime_gate_mean_ratio_max))
        and (p90_ratio is None or p90_ratio <= float(runtime_gate_p90_ratio_max)),
        "baseline_mean_case_time_s": mean_a,
        "candidate_mean_case_time_s": mean_b,
        "baseline_p90_case_time_s": p90_a,
        "candidate_p90_case_time_s": p90_b,
        "mean_ratio_b_over_a": mean_ratio,
        "p90_ratio_b_over_a": p90_ratio,
        "mean_delta_seconds_b_minus_a": mean_delta_seconds,
        "passed_abs_mean_delta": mean_delta_seconds <= float(runtime_gate_abs_mean_increase_max_seconds),
        "distance_from_threshold": {
            "mean_ratio": round(float(runtime_gate_mean_ratio_max) - (mean_ratio or 0), 6) if mean_ratio is not None else None,
            "p90_ratio": round(float(runtime_gate_p90_ratio_max) - (p90_ratio or 0), 6) if p90_ratio is not None else None,
            "mean_delta_seconds": round(float(runtime_gate_abs_mean_increase_max_seconds) - mean_delta_seconds, 6),
        },
        "rule": (
            f"mean_ratio <= {float(runtime_gate_mean_ratio_max)}, "
            f"p90_ratio <= {float(runtime_gate_p90_ratio_max)}, "
            f"and mean_delta_seconds <= {float(runtime_gate_abs_mean_increase_max_seconds)}"
        ),
    }
    runtime_gate["passed"] = bool(runtime_gate["passed"]) and bool(runtime_gate["passed_abs_mean_delta"])

    tool_failure_delta = failure_b - failure_a
    degraded_rate_delta = degraded_rate_b - degraded_rate_a

    max_tf_delta = float(stability_gate_tool_failure_delta_max)
    max_deg_delta = float(stability_gate_degraded_rate_delta_max)

    stability_gate = {
        "name": "stability_gate",
        "passed": tool_failure_delta <= max_tf_delta
        and degraded_rate_delta <= max_deg_delta,
        "baseline_avg_tool_failure_signals": failure_a,
        "candidate_avg_tool_failure_signals": failure_b,
        "tool_failure_delta_b_minus_a": tool_failure_delta,
        "baseline_degraded_case_rate": degraded_rate_a,
        "candidate_degraded_case_rate": degraded_rate_b,
        "degraded_rate_delta_b_minus_a": degraded_rate_delta,
        "n_pairs": n_pairs,
        "tier": tier,
        "thresholds": {"max_tool_failure_delta": max_tf_delta, "max_degraded_rate_delta": max_deg_delta},
        "distance_from_threshold": {
            "tool_failure_delta": round(max_tf_delta - tool_failure_delta, 6),
            "degraded_rate_delta": round(max_deg_delta - degraded_rate_delta, 6),
        },
        "rule": f"tool_failure_delta <= {max_tf_delta} and degraded_rate_delta <= {max_deg_delta}",
    }

    max_catastrophic = int(max(0, tail_risk_max_catastrophic_regressions))
    tail_risk_passed = (catastrophic_regression_count <= max_catastrophic) and (
        True
    )
    tail_risk_gate = {
        "name": "tail_risk_gate",
        "enabled": bool(tail_risk_gate_enabled),
        "passed": bool(tail_risk_passed) if bool(tail_risk_gate_enabled) else True,
        "catastrophic_regression_count": catastrophic_regression_count,
        "catastrophic_threshold": catastrophic_threshold,
        "max_catastrophic_regressions": max_catastrophic,
        "worst_case_delta": worst_case_delta,
        "worst_case_delta_min": min_worst_case_delta,
        "rule": (
            f"catastrophic_regression_count <= {max_catastrophic}"
        ),
    }

    utility_components = {
        "mean_delta": mean_delta,
        "runtime_mean_penalty": 0.15 * max(0.0, (mean_ratio or 1.0) - 1.0),
        "runtime_p90_penalty": 0.10 * max(0.0, (p90_ratio or 1.0) - 1.0),
        "tool_failure_penalty": 0.20 * max(0.0, tool_failure_delta),
        "degraded_rate_penalty": 0.20 * max(0.0, degraded_rate_delta),
    }
    utility_score = (
        utility_components["mean_delta"]
        - utility_components["runtime_mean_penalty"]
        - utility_components["runtime_p90_penalty"]
        - utility_components["tool_failure_penalty"]
        - utility_components["degraded_rate_penalty"]
    )

    gates = [quality_gate, runtime_gate, stability_gate, tail_risk_gate]
    all_passed = all(bool(gate.get("passed")) for gate in gates)
    failed_gates = [str(gate.get("name")) for gate in gates if not bool(gate.get("passed"))]

    return {
        "enabled": bool(enabled),
        "all_passed": bool(all_passed),
        "failed_gates": failed_gates,
        "n_pairs": n_pairs,
        "tier": tier,
        "gates": {gate["name"]: gate for gate in gates},
        "ratios": {
            "mean_case_time_ratio": mean_ratio,
            "p90_case_time_ratio": p90_ratio,
        },
        "delta_stats": {
            "mean_delta": mean_delta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "win_rate": win_rate,
            "p10_delta": p10_delta,
            "worst_case_delta": worst_case_delta,
        },
        "deltas": {
            "tool_failure_delta": tool_failure_delta,
            "degraded_rate_delta": degraded_rate_delta,
            "mean_case_time_delta_seconds": mean_delta_seconds,
            "catastrophic_regression_count": catastrophic_regression_count,
            "catastrophic_threshold": catastrophic_threshold,
        },
        "utility": {
            "score": float(round(utility_score, 6)),
            "components": {k: float(round(v, 6)) for k, v in utility_components.items()},
            "rule": (
                "mean_delta - 0.15*max(0, mean_ratio-1) - 0.10*max(0, p90_ratio-1) "
                "- 0.20*max(0, tool_failure_delta) - 0.20*max(0, degraded_rate_delta)"
            ),
        },
    }


def _evaluate_holdout_confirmation(
    delta_stats: dict[str, Any],
    *,
    holdout_winrate_min: float = HOLDOUT_GATE_MIN_WIN_RATE,
    holdout_min_mean_delta: float = -0.10,
    holdout_catastrophic_threshold: float = TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD,
    holdout_max_catastrophic_regressions: int = 1,
) -> dict[str, Any]:
    mean_delta = float(delta_stats.get("mean_delta", 0.0) or 0.0)
    ci_lower = float(delta_stats.get("ci_lower", mean_delta) or mean_delta)
    ci_upper = float(delta_stats.get("ci_upper", mean_delta) or mean_delta)
    deltas = list(delta_stats.get("deltas") or [])
    wins = sum(1 for delta in deltas if float(delta) > 0.0)
    pairs = len(deltas)
    win_rate = (float(wins) / float(pairs)) if pairs else 0.0
    catastrophic_threshold = float(holdout_catastrophic_threshold)
    catastrophic_regression_count = sum(1 for delta in deltas if float(delta) <= catastrophic_threshold)
    checks = {
        "win_rate_minimum": win_rate >= float(holdout_winrate_min),
        "mean_delta_floor": mean_delta >= float(holdout_min_mean_delta),
        "catastrophic_limit": catastrophic_regression_count <= int(max(0, holdout_max_catastrophic_regressions)),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "passed": passed,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "wins": wins,
        "pairs": pairs,
        "win_rate": win_rate,
        "win_rate_min": float(holdout_winrate_min),
        "mean_delta_min": float(holdout_min_mean_delta),
        "catastrophic_threshold": catastrophic_threshold,
        "max_catastrophic_regressions": int(max(0, holdout_max_catastrophic_regressions)),
        "catastrophic_regression_count": catastrophic_regression_count,
        "checks": checks,
        "rule": (
            f"win_rate >= {float(holdout_winrate_min)} and "
            f"mean_delta >= {float(holdout_min_mean_delta)} and "
            f"catastrophic_regression_count <= {int(max(0, holdout_max_catastrophic_regressions))}"
        ),
    }


def _build_rejection_reason_codes(
    *,
    gate_failure_reasons: list[str],
    gate_results: dict[str, Any],
    candidate_exists: bool,
    evaluation_mode: str,
) -> list[dict[str, Any]]:
    reasons = list(gate_failure_reasons or [])
    if not reasons:
        return []

    severity_map = {
        "no_candidate_changes": "high",
        "precheck_rejected": "high",
        "quality_gate": "high",
        "tail_risk_gate": "high",
        "runtime_gate": "medium",
        "stability_gate": "medium",
        "holdout_confirmation_gate": "high",
    }
    evidence_map = {
        "quality_gate": "candidate_gate_results.gates.quality_gate",
        "tail_risk_gate": "candidate_gate_results.gates.tail_risk_gate",
        "runtime_gate": "candidate_gate_results.gates.runtime_gate",
        "stability_gate": "candidate_gate_results.gates.stability_gate",
        "holdout_confirmation_gate": "selection_stats.holdout_confirmation",
        "no_candidate_changes": "phase_b_eval_meta.evaluation_mode",
        "precheck_rejected": "selection_stats.precheck_trace",
    }
    out: list[dict[str, Any]] = []
    for index, code in enumerate(reasons):
        out.append(
            {
                "code": str(code),
                "severity": severity_map.get(str(code), "medium"),
                "rank": index + 1,
                "evidence": evidence_map.get(str(code), "metadata"),
                "candidate_exists": bool(candidate_exists),
                "evaluation_mode": str(evaluation_mode),
            }
        )

    if not candidate_exists and all(item.get("code") != "no_candidate_changes" for item in out):
        out.insert(
            0,
            {
                "code": "candidate_not_evaluated",
                "severity": "high",
                "rank": 1,
                "evidence": "phase_b_eval_meta.evaluation_mode",
                "candidate_exists": False,
                "evaluation_mode": str(evaluation_mode),
            },
        )
        for idx, item in enumerate(out, start=1):
            item["rank"] = idx
    return out


_GRADE_WEIGHTS: dict[str, float] = {
    "prompt_alignment_score": 1.0,
    "factuality_score": 1.2,
    "clarity_score": 1.0,
    "helpfulness_score": 1.0,
    "safety_score": 0.2,
    "tool_usage_score": 0.8,
    "format_quality_score": 0.9,
    "engagement_score": 0.7,
    "citation_quality_score": 1.0,
}


def _weighted_aggregate(scores: dict[str, int | float]) -> float:
    """Compute weighted average across grading dimensions."""
    total_weight = 0.0
    total_value = 0.0
    for key, weight in _GRADE_WEIGHTS.items():
        value = scores.get(key)
        if value is not None:
            total_weight += weight
            total_value += float(value) * weight
    if total_weight == 0.0:
        return 0.0
    return total_value / total_weight


class GenericGradeSchema(BaseModel):
    prompt_alignment_score: int = Field(ge=1, le=10)
    factuality_score: int = Field(ge=1, le=10)
    clarity_score: int = Field(ge=1, le=10)
    helpfulness_score: int = Field(ge=1, le=10)
    safety_score: int = Field(ge=1, le=10)
    tool_usage_score: int = Field(ge=1, le=10)
    format_quality_score: int = Field(ge=1, le=10)
    engagement_score: int = Field(ge=1, le=10)
    citation_quality_score: int = Field(ge=1, le=10)
    major_issues: str
    strengths: str


class BlockAnalysisSchema(BaseModel):
    block_id: str
    generic_issue_score: int = Field(ge=1, le=10)
    criteria_misalignment_score: int = Field(ge=1, le=10)
    need_fix: bool
    analysis: str
    what_to_fix: str
    failure_mode: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None
    linked_theme_ids: list[str] = Field(default_factory=list)
    recommended_fix_patterns: list[str] = Field(default_factory=list)
    likely_issue_points: list[str] = Field(default_factory=list)


class PipelineAnalysisReportSchema(BaseModel):
    overall_recommendations: str
    block_analyses: list[BlockAnalysisSchema]


class HighLevelRCAThemeSchema(BaseModel):
    theme_id: str
    description: str
    evidence_case_ids: list[str] = Field(default_factory=list)
    implicated_blocks: list[str] = Field(default_factory=list)
    severity: int = Field(ge=1, le=10)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    recommended_fix_patterns: list[str] = Field(default_factory=list)


class HighLevelRCAReportSchema(BaseModel):
    summary: str
    themes: list[HighLevelRCAThemeSchema]


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


def _coerce_schema_result(
    value: Any,
    schema: type[BaseModel],
    *,
    model_name: str,
) -> BaseModel:
    if isinstance(value, schema):
        return value

    if isinstance(value, BaseModel):
        try:
            return schema.model_validate(value.model_dump())
        except Exception:
            pass

    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                return schema.model_validate_json(raw)
            except Exception:
                pass
            json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if json_match:
                try:
                    return schema.model_validate_json(json_match.group(0))
                except Exception:
                    pass

    fallback_builder = getattr(utility_module, "_graceful_schema_fallback", None)
    if callable(fallback_builder):
        fallback = fallback_builder(
            schema,
            f"[Graceful degradation] Invalid structured output from model={model_name}.",
        )
        if isinstance(fallback, schema):
            return fallback
        if isinstance(fallback, BaseModel):
            try:
                return schema.model_validate(fallback.model_dump())
            except Exception:
                pass

    raise TypeError(f"Unable to coerce output into schema {schema.__name__}")


def _generate_with_oss_fallback(
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.2,
) -> tuple[BaseModel, str]:
    """Try oss120b first (fast-fail), then nemotron. OSS has 8k token limit."""
    attempts: list[tuple[str, BaseModel]] = []
    model_order: list[str] = ["oss120b", "nemotron"]
    if _should_skip_oss_for_prompt(prompt):
        logger.debug("Skipping oss120b for oversized prompt (%d est. tokens); using nemotron",
                      _estimated_tokens_from_text(prompt))
        model_order = ["nemotron"]

    for model in model_order:
        # OSS: try once with very short timeout; nemotron: standard retries
        retries = 1 if model == "oss120b" else 5
        max_retry_wait = 5.0 if model == "oss120b" else 30.0
        output_raw = generate_text(
            prompt=prompt,
            model=model,
            schema=schema,
            temperature=temperature,
            retries=retries,
            max_total_retry_wait=max_retry_wait,
        )
        if ENABLE_RESOURCE_TELEMETRY_V3 and _ACTIVE_RESOURCE_TELEMETRY is not None:
            try:
                _ACTIVE_RESOURCE_TELEMETRY.record_llm_call(
                    tokens_in=_estimated_tokens_from_text(prompt),
                    tokens_out=_estimated_tokens_from_text(_safe_json_dumps(output_raw, max_chars=24000)),
                )
            except Exception:
                pass
        try:
            output = _coerce_schema_result(output_raw, schema, model_name=model)
        except Exception as exc:
            logger.warning(
                "Schema coercion failed for model=%s schema=%s: %s",
                model,
                schema.__name__,
                exc,
            )
            continue
        attempts.append((model, output))
        if not _contains_degraded_marker(output):
            return output, model

    if not attempts:
        fallback = _coerce_schema_result(
            "[Graceful degradation] All schema coercions failed across fallback models.",
            schema,
            model_name="fallback",
        )
        return fallback, "fallback"

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
    if ENABLE_TOOL_PROMPT_TEMPLATES:
        try:
            tool_descriptors = get_tool_prompt_descriptors()
            if isinstance(tool_descriptors, dict):
                for block_id, descriptor in tool_descriptors.items():
                    details[str(block_id)] = copy.deepcopy(descriptor)
        except Exception as exc:
            logger.warning(f"Unable to load tool prompt descriptors: {exc}")
    return details


class TrainingCaseSchema(BaseModel):
    id: str
    category: str
    difficulty: str
    prompt: str
    validation: str
    answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    strictness_level: str = "medium"
    anti_overfit_markers: list[str] = Field(default_factory=list)


def _normalize_case_tags(case: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    category = str(case.get("category") or "").strip().lower()
    difficulty = str(case.get("difficulty") or "").strip().lower()
    if category:
        tags.add(category)
    if difficulty:
        tags.add(difficulty)

    prompt_text = str(case.get("prompt") or "").lower()
    validation_text = str(case.get("validation") or "").lower()
    combined = f"{prompt_text}\n{validation_text}"
    marker_map = {
        "citation": "citation_quality",
        "source": "citation_quality",
        "proof": "formal_reasoning",
        "counterexample": "formal_reasoning",
        "python": "python_required",
        "code": "coding",
        "table": "formatting",
        "json": "formatting",
        "timeout": "runtime_sensitivity",
        "deterministic": "deterministic_answer",
        "must not": "strict_constraints",
        "must include": "strict_constraints",
        "exactly": "strict_constraints",
    }
    for token, tag in marker_map.items():
        if token in combined:
            tags.add(tag)
    existing = case.get("tags")
    if isinstance(existing, list):
        for item in existing:
            item_text = str(item).strip().lower()
            if item_text:
                tags.add(item_text)
    return sorted(tags)


def _infer_strictness_level(validation_text: str) -> str:
    normalized = str(validation_text or "").lower()
    hard_markers = (
        "must not",
        "must include",
        "exactly",
        "do not",
        "strict",
        "not exceed",
        "greater than",
    )
    medium_markers = (
        "should",
        "include",
        "compare",
        "explain",
        "discuss",
    )
    hard_count = sum(1 for marker in hard_markers if marker in normalized)
    medium_count = sum(1 for marker in medium_markers if marker in normalized)
    if hard_count >= 4:
        return "critical"
    if hard_count >= 2:
        return "high"
    if medium_count >= 2:
        return "medium"
    return "low"


def _normalize_training_case(raw_case: dict[str, Any], index: int) -> dict[str, Any]:
    case = dict(raw_case or {})
    case_id = str(case.get("id") or f"case-{index:04d}").strip()
    category = str(case.get("category") or "uncategorized").strip().lower()
    difficulty = str(case.get("difficulty") or "medium").strip().lower()
    prompt = str(case.get("prompt") or "").strip()
    validation = str(case.get("validation") or "").strip()
    answer = case.get("answer")
    answer_text = str(answer).strip() if answer is not None else None
    case["id"] = case_id
    case["category"] = category
    case["difficulty"] = difficulty
    case["prompt"] = prompt
    case["validation"] = validation
    case["answer"] = answer_text if answer_text else None
    case["tags"] = _normalize_case_tags(case)
    expected_tools = case.get("expected_tools")
    if not isinstance(expected_tools, list):
        expected_tools = []
    case["expected_tools"] = [str(item).strip() for item in expected_tools if str(item).strip()]
    failure_modes = case.get("failure_modes")
    if not isinstance(failure_modes, list):
        failure_modes = []
    case["failure_modes"] = [str(item).strip() for item in failure_modes if str(item).strip()]
    anti_overfit_markers = case.get("anti_overfit_markers")
    if not isinstance(anti_overfit_markers, list):
        anti_overfit_markers = []
    if not anti_overfit_markers:
        anti_overfit_markers = [f"id::{case_id}", f"category::{category}"]
    case["anti_overfit_markers"] = [
        str(item).strip() for item in anti_overfit_markers if str(item).strip()
    ]
    strictness = str(case.get("strictness_level") or "").strip().lower()
    case["strictness_level"] = strictness or _infer_strictness_level(validation)
    validated = TrainingCaseSchema.model_validate(case)
    return validated.model_dump()


def _dataset_balance_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    strictness_counts: dict[str, int] = {}
    for case in cases:
        category = str(case.get("category") or "uncategorized")
        difficulty = str(case.get("difficulty") or "unknown")
        strictness = str(case.get("strictness_level") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        strictness_counts[strictness] = strictness_counts.get(strictness, 0) + 1
    return {
        "cases_count": len(cases),
        "categories_count": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "strictness_counts": dict(sorted(strictness_counts.items())),
    }


def load_test_cases_with_diagnostics(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Training dataset must be a list: {path}")

    normalized_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    dropped_cases = 0
    derived_field_counts: dict[str, int] = {
        "tags": 0,
        "expected_tools": 0,
        "failure_modes": 0,
        "strictness_level": 0,
        "anti_overfit_markers": 0,
    }
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            dropped_cases += 1
            continue
        for field in derived_field_counts:
            if field not in item:
                derived_field_counts[field] += 1
        case = _normalize_training_case(item, idx)
        case_id = _case_identifier(case)
        if case_id in seen_ids:
            duplicate_ids.append(case_id)
            case["id"] = f"{case['id']}__dup_{idx}"
            case_id = _case_identifier(case)
        seen_ids.add(case_id)
        normalized_cases.append(case)

    diagnostics = {
        "dataset_schema_version": DEFAULT_DATASET_SCHEMA_VERSION,
        "input_cases_count": len(data),
        "normalized_cases_count": len(normalized_cases),
        "dropped_cases_count": dropped_cases,
        "duplicate_ids": duplicate_ids[:20],
        "derived_v2_field_counts": derived_field_counts,
        **_dataset_balance_summary(normalized_cases),
    }
    return normalized_cases, diagnostics


def load_test_cases(path: str) -> list[dict[str, Any]]:
    cases, _ = load_test_cases_with_diagnostics(path)
    return cases


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
        events_archive_path: str | None,
        run_id: str,
        epochs_total: int,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.epochs_total = int(max(0, epochs_total))
        self.seq = 0
        self._last_case_activity_unix: float | None = None
        self._last_stalled_warning_unix: float = 0.0
        self.stalled_warning_threshold_seconds = max(60.0, float(STALL_WARNING_THRESHOLD_SECONDS))
        self._run_started_perf = time.perf_counter()
        self._phase_started_perf = self._run_started_perf
        self._last_phase = "init"
        self._emit_latency_ms_total = 0.0
        self._write_status_latency_ms_total = 0.0
        self._write_event_latency_ms_total = 0.0
        self._phase_snapshot_every = 25
        self.status_path = Path(status_path)
        self.events_path = Path(events_path)
        self.events_archive_path = Path(events_archive_path) if events_archive_path else None
        self.status: dict[str, Any] = {
            "run_id": run_id,
            "state": "idle",
            "started_at": _iso_now(),
            "started_at_unix": round(time.time(), 3),
            "updated_at": _iso_now(),
            "updated_at_unix": round(time.time(), 3),
            "epoch_current": 0,
            "epochs_total": self.epochs_total,
            "phase": "init",
            "step": "init",
            "message": "",
            "events_count": 0,
            "last_event_type": None,
            "current_case": None,
            "active_call": None,
            "stalled_warning_count": 0,
            "elapsed_seconds": 0.0,
            "phase_elapsed_seconds": 0.0,
            "metrics": {},
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "telemetry_mode": "dense_compressed",
            "paths": {
                "status_path": str(self.status_path),
                "events_path": str(self.events_path),
                "events_archive_path": str(self.events_archive_path) if self.events_archive_path else None,
                "log_path": str(Path("logs/prompt_training.log")),
            },
        }

        if not self.enabled:
            return

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_archive_path is not None:
            self.events_archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Start each run with a fresh event log for clean streaming.
        self.events_path.write_text("")
        if self.events_archive_path is not None:
            self.events_archive_path.write_text("")
        self._write_status()

    def _write_status(self) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        tmp_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.status, indent=2))
        tmp_path.replace(self.status_path)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._write_status_latency_ms_total += elapsed_ms

    def _append_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        if self.events_archive_path is not None:
            with self.events_archive_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._write_event_latency_ms_total += elapsed_ms

    def _compress_payload(self, payload: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        base_payload = payload or {}
        try:
            payload_text = _safe_json_dumps(base_payload)
        except Exception:
            payload_text = str(base_payload)
        payload_bytes = len(payload_text.encode("utf-8", errors="ignore"))
        payload_hash = hashlib.sha256(payload_text.encode("utf-8", errors="ignore")).hexdigest()
        payload_ref_id = f"payload:{payload_hash[:16]}"
        compressed = False
        if payload_bytes > 12000:
            compressed = True
            base_payload = {
                "_compressed": True,
                "payload_ref_id": payload_ref_id,
                "payload_sha256": payload_hash,
                "payload_bytes": payload_bytes,
                "payload_preview": _prune_for_rca(
                    payload or {},
                    max_depth=6,
                    max_items=40,
                    max_text_chars=300,
                ),
            }
        return base_payload, {
            "payload_ref_id": payload_ref_id,
            "payload_sha256": payload_hash,
            "payload_bytes": payload_bytes,
            "payload_compressed": compressed,
        }

    def _emit_phase_profile_snapshot(self, *, phase: str, epoch_current: int, timestamp_unix: float) -> None:
        if not self.enabled:
            return
        self.seq += 1
        elapsed_seconds = max(0.0, time.perf_counter() - self._run_started_perf)
        events_per_min = (float(self.seq) / max(elapsed_seconds / 60.0, 1e-6))
        avg_emit_latency = self._emit_latency_ms_total / max(float(self.seq), 1.0)
        avg_write_latency = (
            (self._write_status_latency_ms_total + self._write_event_latency_ms_total)
            / max(float(self.seq), 1.0)
        )
        snapshot_event = {
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": _iso_now(),
            "timestamp_unix": round(timestamp_unix, 3),
            "event_type": "phase_profile_snapshot",
            "phase": phase,
            "step": "telemetry_profile",
            "message": f"Telemetry snapshot for phase={phase}",
            "epoch_current": int(epoch_current),
            "epochs_total": self.epochs_total,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "phase_elapsed_seconds": round(max(0.0, time.perf_counter() - self._phase_started_perf), 3),
            "payload": {
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "events_count": int(self.seq),
                "events_per_min": round(events_per_min, 3),
                "avg_emit_latency_ms": round(avg_emit_latency, 4),
                "avg_tracker_write_latency_ms": round(avg_write_latency, 4),
                "stream_lag_ms": 0.0,
            },
            "current_case": self.status.get("current_case"),
            "metrics": {},
            "active_call": self._active_call_snapshot(timestamp_unix),
        }
        self._append_event(snapshot_event)
        self.status["events_count"] = self.seq
        self.status["last_event_type"] = "phase_profile_snapshot"
        self.status["updated_at"] = _iso_now()
        self.status["updated_at_unix"] = round(timestamp_unix, 3)
        self._write_status()

    def _active_call_snapshot(self, now_unix: float | None = None) -> dict[str, Any] | None:
        active_call = self.status.get("active_call")
        if not isinstance(active_call, dict):
            return None

        snapshot = dict(active_call)
        start_unix = snapshot.get("started_at_unix")
        if isinstance(start_unix, (int, float)):
            current = float(now_unix if now_unix is not None else time.time())
            snapshot["elapsed_call_seconds"] = round(max(0.0, current - float(start_unix)), 3)
        return snapshot

    def set_active_call(
        self,
        block_id: str,
        *,
        attempt: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        now_iso = _iso_now()
        now_unix = round(time.time(), 3)
        active_call = {
            "block_id": str(block_id),
            "attempt": int(attempt) if isinstance(attempt, int) else None,
            "metadata": metadata or {},
            "started_at": now_iso,
            "started_at_unix": now_unix,
        }
        self.status["active_call"] = active_call
        self.status["updated_at"] = now_iso
        self.status["updated_at_unix"] = now_unix
        self._write_status()

    def clear_active_call(self) -> None:
        if not self.enabled:
            return
        if self.status.get("active_call") is None:
            return
        self.status["active_call"] = None
        self.status["updated_at"] = _iso_now()
        self.status["updated_at_unix"] = round(time.time(), 3)
        self._write_status()

    @contextmanager
    def active_call(
        self,
        block_id: str,
        *,
        attempt: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.set_active_call(block_id, attempt=attempt, metadata=metadata)
        try:
            yield
        finally:
            self.clear_active_call()

    def _emit_stalled_warning_if_needed(
        self,
        *,
        now_unix: float,
        phase: str,
        epoch_current: int,
    ) -> None:
        if not self.enabled:
            return
        if self._last_case_activity_unix is None:
            return
        stall_seconds = float(now_unix - self._last_case_activity_unix)
        if stall_seconds < self.stalled_warning_threshold_seconds:
            return
        if (
            self._last_stalled_warning_unix
            and (now_unix - self._last_stalled_warning_unix) < self.stalled_warning_threshold_seconds
        ):
            return

        self.seq += 1
        warning_event = {
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": _iso_now(),
            "timestamp_unix": round(now_unix, 3),
            "event_type": "stalled_warning",
            "phase": phase,
            "step": "watchdog",
            "message": (
                f"No case completion activity for {int(stall_seconds)}s; intervention may be required."
            ),
            "epoch_current": int(epoch_current),
            "epochs_total": self.epochs_total,
            "elapsed_seconds": round(max(0.0, time.perf_counter() - self._run_started_perf), 3),
            "phase_elapsed_seconds": round(max(0.0, time.perf_counter() - self._phase_started_perf), 3),
            "payload": {
                "threshold_seconds": int(self.stalled_warning_threshold_seconds),
                "stalled_seconds": int(stall_seconds),
                "active_call": self._active_call_snapshot(now_unix),
            },
            "current_case": self.status.get("current_case"),
            "metrics": {},
            "active_call": self._active_call_snapshot(now_unix),
        }
        self._append_event(warning_event)
        self._last_stalled_warning_unix = now_unix
        self.status["events_count"] = self.seq
        self.status["last_event_type"] = "stalled_warning"
        self.status["stalled_warning_count"] = int(self.status.get("stalled_warning_count", 0)) + 1
        self._write_status()
        logger.warning(
            f"[training:stalled_warning] epoch={epoch_current}/{self.epochs_total} "
            f"phase={phase} stalled_seconds={int(stall_seconds)}"
        )

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
        started_perf = time.perf_counter()
        timestamp = _iso_now()
        timestamp_unix = round(time.time(), 3)
        compact_payload, payload_diag = self._compress_payload(payload or {})
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
            "timestamp_unix": timestamp_unix,
            "event_type": event_type,
            "phase": phase,
            "step": step,
            "message": message,
            "epoch_current": int(epoch_current or self.status.get("epoch_current", 0)),
            "epochs_total": self.epochs_total,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "phase_elapsed_seconds": round(phase_elapsed_seconds, 3),
            "payload": compact_payload,
            "current_case": current_case,
            "metrics": metrics or {},
            "active_call": self._active_call_snapshot(timestamp_unix),
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "payload_ref_id": payload_diag.get("payload_ref_id"),
            "payload_sha256": payload_diag.get("payload_sha256"),
            "payload_bytes": payload_diag.get("payload_bytes"),
            "payload_compressed": bool(payload_diag.get("payload_compressed")),
        }

        self.status["updated_at"] = timestamp
        self.status["updated_at_unix"] = timestamp_unix
        self.status["phase"] = phase
        self.status["step"] = step
        self.status["message"] = message
        self.status["events_count"] = self.seq
        self.status["last_event_type"] = event_type
        self.status["active_call"] = self._active_call_snapshot(timestamp_unix)
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

        if event_type in {"case_started", "case_completed"}:
            self._last_case_activity_unix = timestamp_unix

        emit_elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
        self._emit_latency_ms_total += emit_elapsed_ms
        events_per_min = (float(self.seq) / max(elapsed_seconds / 60.0, 1e-6))
        avg_emit_latency_ms = self._emit_latency_ms_total / max(float(self.seq), 1.0)
        avg_tracker_write_latency_ms = (
            (self._write_status_latency_ms_total + self._write_event_latency_ms_total)
            / max(float(self.seq), 1.0)
        )
        merged_metrics = dict(self.status.get("metrics", {}))
        merged_metrics.update(
            {
                "events_per_min": round(events_per_min, 3),
                "avg_emit_latency_ms": round(avg_emit_latency_ms, 4),
                "avg_tracker_write_latency_ms": round(avg_tracker_write_latency_ms, 4),
                "stream_lag_ms": 0.0,
            }
        )
        self.status["metrics"] = merged_metrics

        self._write_status()
        self._append_event(event)
        if self.seq > 0 and (self.seq % int(max(1, self._phase_snapshot_every)) == 0):
            self._emit_phase_profile_snapshot(
                phase=phase,
                epoch_current=int(event.get("epoch_current") or 0),
                timestamp_unix=timestamp_unix,
            )
        if self.status.get("state") == "running" and event_type not in {"stalled_warning", "run_completed", "run_error"}:
            self._emit_stalled_warning_if_needed(
                now_unix=timestamp_unix,
                phase=phase,
                epoch_current=int(event.get("epoch_current") or 0),
            )
        logger.info(
            f"[training:{event_type}] epoch={event.get('epoch_current')}/{self.epochs_total} "
            f"phase={phase} step={step} msg={message}"
        )

    def complete(self, metrics: dict[str, Any] | None = None) -> None:
        self.clear_active_call()
        self.emit(
            "run_completed",
            phase="complete",
            step="finished",
            message="Training completed successfully",
            epoch_current=self.epochs_total,
            metrics=metrics or {},
            current_case=None,
        )
        if not self.enabled:
            return
        if self.events_archive_path is None or not self.events_archive_path.exists():
            return
        try:
            gz_path = self.events_archive_path.with_suffix(self.events_archive_path.suffix + ".gz")
            with self.events_archive_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                dst.write(src.read())
            self.status["paths"]["events_archive_gzip_path"] = str(gz_path)
            self.status["updated_at"] = _iso_now()
            self.status["updated_at_unix"] = round(time.time(), 3)
            self._write_status()
        except Exception as exc:
            logger.warning("Could not gzip training event archive: %s", exc)

    def fail(self, error_message: str, epoch_current: int | None = None) -> None:
        self.clear_active_call()
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
    short_scores = {
        "prompt_alignment_score": 1,
        "factuality_score": 1,
        "clarity_score": 2,
        "helpfulness_score": 1,
        "safety_score": 5,
        "tool_usage_score": 1,
        "format_quality_score": 1,
        "engagement_score": 1,
        "citation_quality_score": 1,
        "major_issues": major,
        "strengths": "",
        "grading_mode": "short_circuit_failure",
    }
    short_scores["aggregate_score"] = _weighted_aggregate(short_scores)
    return short_scores


def _normalize_match_text(value: str) -> str:
    lowered = str(value or "").lower()
    lowered = re.sub(r"[^a-z0-9\.\-\+\*/=]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _extract_numeric_tokens(value: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?(?:e[+\-]?\d+)?", str(value or "").lower())


def _deterministic_answer_score(case: dict[str, Any], run_output: dict[str, Any]) -> tuple[float | None, str]:
    answer = case.get("answer")
    if answer is None:
        return None, "no_answer"

    answer_text = str(answer).strip()
    if not answer_text:
        return None, "empty_answer"

    response_text = str(run_output.get("response", "")).strip()
    if not response_text:
        return 1.0, "empty_response"

    answer_norm = _normalize_match_text(answer_text)
    response_norm = _normalize_match_text(response_text)

    if answer_norm and answer_norm == response_norm:
        return 10.0, "exact_match"
    if answer_norm and answer_norm in response_norm:
        return 9.0, "substring_match"

    answer_numbers = _extract_numeric_tokens(answer_text)
    response_numbers = set(_extract_numeric_tokens(response_text))
    if answer_numbers:
        matched = sum(1 for item in answer_numbers if item in response_numbers)
        ratio = matched / max(len(answer_numbers), 1)
        if ratio >= 1.0:
            return 9.0, "numeric_full_match"
        if ratio >= 0.5:
            return 6.5, "numeric_partial_match"
        return 2.5, "numeric_mismatch"

    answer_tokens = set(re.findall(r"[a-z0-9]+", answer_norm))
    response_tokens = set(re.findall(r"[a-z0-9]+", response_norm))
    if not answer_tokens:
        return None, "no_tokens"
    overlap_ratio = len(answer_tokens.intersection(response_tokens)) / len(answer_tokens)
    if overlap_ratio >= 0.85:
        return 8.5, "token_overlap_high"
    if overlap_ratio >= 0.6:
        return 6.0, "token_overlap_medium"
    if overlap_ratio >= 0.35:
        return 4.0, "token_overlap_low"
    return 2.0, "token_overlap_minimal"


def _reliability_penalty(run_output: dict[str, Any], *, timed_out: bool) -> dict[str, Any]:
    tool_metrics = _extract_case_tool_metrics(run_output)
    penalty_total = 0.0
    breakdown: dict[str, float] = {}

    if bool(run_output.get("degraded_mode_active")):
        breakdown["degraded_mode_active"] = 0.5
        penalty_total += breakdown["degraded_mode_active"]
    if timed_out:
        breakdown["case_timed_out"] = 2.0
        penalty_total += breakdown["case_timed_out"]

    tool_failure_signals = int(tool_metrics.get("tool_failure_signals_count", 0))
    repeated_signals = max(0, tool_failure_signals - 1)
    if repeated_signals > 0:
        breakdown["repeated_tool_failure_signals"] = min(1.5, 0.15 * repeated_signals)
        penalty_total += breakdown["repeated_tool_failure_signals"]

    python_failures = int(tool_metrics.get("python_exec_failures", 0))
    if python_failures >= 2:
        breakdown["repeated_python_exec_failures"] = min(1.0, 0.2 * (python_failures - 1))
        penalty_total += breakdown["repeated_python_exec_failures"]

    return {
        "total": round(penalty_total, 6),
        "breakdown": breakdown,
    }


def _postprocess_grade(
    *,
    case: dict[str, Any],
    run_output: dict[str, Any],
    grade: dict[str, Any],
    timed_out: bool = False,
) -> dict[str, Any]:
    normalized_grade = dict(grade)
    base_aggregate = float(normalized_grade.get("aggregate_score", _weighted_aggregate(normalized_grade)))
    deterministic_score, deterministic_mode = _deterministic_answer_score(case, run_output)

    blended_aggregate = base_aggregate
    if deterministic_score is not None:
        blended_aggregate = (0.75 * base_aggregate) + (0.25 * float(deterministic_score))

    penalty_info = _reliability_penalty(run_output, timed_out=timed_out)
    final_aggregate = _clamp_score(blended_aggregate - float(penalty_info.get("total", 0.0)))

    normalized_grade["aggregate_score_base"] = round(base_aggregate, 6)
    normalized_grade["aggregate_score_pre_penalty"] = round(blended_aggregate, 6)
    normalized_grade["deterministic_answer_score"] = (
        round(float(deterministic_score), 6) if deterministic_score is not None else None
    )
    normalized_grade["deterministic_answer_mode"] = deterministic_mode
    normalized_grade["reliability_penalty"] = float(penalty_info.get("total", 0.0))
    normalized_grade["reliability_penalty_breakdown"] = penalty_info.get("breakdown", {})
    normalized_grade["infrastructure_degraded"] = float(penalty_info.get("total", 0.0)) >= 2.0
    normalized_grade["aggregate_score"] = round(final_aggregate, 6)
    return normalized_grade


def grade_result(
    case: dict[str, Any],
    run_output: dict[str, Any],
) -> dict[str, Any]:
    prompt = str(case.get("prompt", ""))
    validation = case.get("validation")
    final_response = str(run_output.get("response", ""))
    grading_prompt = (
        "You are a harsh, calibrated grader for AI pipeline responses.\n"
        "Score each dimension strictly from 1 to 10 using these anchors:\n\n"
        "CALIBRATION ANCHORS (use these to set your scale):\n"
        "  9-10: Publishable quality. Expert-level, no meaningful flaws, would pass peer review.\n"
        "  7-8:  Competent with minor gaps. Solid work but has identifiable weaknesses.\n"
        "  5-6:  Partially correct. Contains substantive errors, omissions, or structural problems.\n"
        "  3-4:  Fundamentally flawed. Major errors in logic, facts, or approach.\n"
        "  1-2:  Fails to address the task or is incoherent.\n\n"
        "EXPLICIT PENALTIES (apply before scoring):\n"
        "  - Hallucinated citations or fabricated references: -3 on factuality_score\n"
        "  - Excessive verbosity or filler without substance: -2 on engagement_score\n"
        "  - Missing requested structure (e.g., 'provide a table' but gave prose): -2 on format_quality_score\n"
        "  - Claims presented without evidence or reasoning: -2 on helpfulness_score\n\n"
        f"User Prompt:\n{prompt}\n\n"
        f"Final Response:\n{final_response}\n\n"
        f"Validation Criteria (ground truth when available):\n{validation or 'N/A'}\n\n"
        "Rubric (score each 1-10):\n"
        "- prompt_alignment_score: instruction compliance and scope match\n"
        "- factuality_score: correctness, non-hallucination, and verifiable accuracy\n"
        "- clarity_score: structure, readability, and logical organization\n"
        "- helpfulness_score: usefulness, actionability, and depth of insight\n"
        "- safety_score: policy/safety compliance\n"
        "- tool_usage_score: appropriate and grounded use of tool evidence\n"
        "- format_quality_score: visual layout, markdown formatting, use of headers/lists/tables as appropriate\n"
        "- engagement_score: conciseness, avoidance of filler, intellectual rigor\n"
        "- citation_quality_score: quality and groundedness of references and evidence; penalize fabricated citations heavily\n\n"
        "Be strict. A 'pretty good' response is a 6-7, not an 8-9.\n"
        "Also provide major_issues and strengths."
    )

    graded_output, grading_model = _generate_with_oss_fallback(
        prompt=grading_prompt,
        schema=GenericGradeSchema,
        temperature=0.0,
    )

    payload = graded_output.model_dump()
    payload["aggregate_score"] = _weighted_aggregate(payload)
    payload["grading_model"] = grading_model
    return payload


def _build_rca_evidence_bundle(run_output: dict[str, Any], case_context: dict[str, Any] | None) -> dict[str, Any]:
    compact_run_output = _compact_run_output_for_storage(run_output)
    tool_metrics = _extract_case_tool_metrics(run_output)
    step_coverage = run_output.get("step_coverage")
    summarized_tool_invocations = sorted(
        (
            {"block_id": str(block_id), "count": int(count)}
            for block_id, count in tool_metrics.get("tool_invocations", {}).items()
            if isinstance(count, (int, float))
        ),
        key=lambda item: item["count"],
        reverse=True,
    )[:12]
    summarized_tool_errors = sorted(
        (
            {"block_id": str(block_id), "count": int(count)}
            for block_id, count in tool_metrics.get("tool_errors", {}).items()
            if isinstance(count, (int, float))
        ),
        key=lambda item: item["count"],
        reverse=True,
    )[:12]
    return {
        "case_metadata": {
            "id": (case_context or {}).get("id"),
            "category": (case_context or {}).get("category"),
            "difficulty": (case_context or {}).get("difficulty"),
            "aggregate_score": (case_context or {}).get("aggregate_score"),
            "degraded_mode_active": bool((case_context or {}).get("degraded_mode_active")),
            "timed_out": bool((case_context or {}).get("timed_out")),
        },
        "step_coverage": _prune_for_rca(step_coverage, max_depth=5, max_items=18, max_text_chars=500),
        "tool_failure_signals": _prune_for_rca(run_output.get("tool_failure_signals"), max_depth=4, max_items=12, max_text_chars=260),
        "tool_invocations_summary": summarized_tool_invocations,
        "tool_errors_summary": summarized_tool_errors,
        "degraded_notes": _prune_for_rca(run_output.get("degraded_notes"), max_depth=4, max_items=8, max_text_chars=260),
        "run_output_compact": compact_run_output,
    }


def _dedupe_non_empty_strings(values: list[Any], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if max_items is not None and len(deduped) >= int(max_items):
            break
    return deduped


_RCA_LINKAGE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "case",
    "cases",
    "block",
    "blocks",
    "prompt",
    "prompts",
    "issue",
    "issues",
    "fix",
    "needs",
    "need",
    "mode",
    "analysis",
    "major",
    "minor",
    "should",
    "must",
}


def _text_tokens_for_rca_linkage(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]{3,}", str(text or "").lower())
    return {token for token in tokens if token not in _RCA_LINKAGE_STOPWORDS}


def _link_analyses_to_high_level_themes(
    analyses: list[BlockAnalysisSchema],
    high_level_themes: list[dict[str, Any]] | None,
) -> None:
    if not analyses or not high_level_themes:
        return

    theme_rows: list[dict[str, Any]] = []
    for theme in high_level_themes:
        if not isinstance(theme, dict):
            continue
        theme_id = str(theme.get("theme_id") or "").strip()
        if not theme_id:
            continue
        description = str(theme.get("description") or "")
        implicated_blocks = [str(item) for item in (theme.get("implicated_blocks") or []) if str(item).strip()]
        fix_patterns = _dedupe_non_empty_strings(list(theme.get("recommended_fix_patterns") or []), max_items=8)
        theme_text = " ".join(
            [
                description,
                " ".join(fix_patterns),
                " ".join(str(item) for item in (theme.get("evidence_case_ids") or [])),
            ]
        )
        theme_rows.append(
            {
                "theme_id": theme_id,
                "severity": int(theme.get("severity", 1) or 1),
                "description": description,
                "tokens": _text_tokens_for_rca_linkage(theme_text),
                "implicated_blocks": set(implicated_blocks),
                "fix_patterns": fix_patterns,
            }
        )

    if not theme_rows:
        return

    for analysis in analyses:
        analysis_tokens = _text_tokens_for_rca_linkage(
            " ".join(
                [
                    str(analysis.analysis or ""),
                    str(analysis.what_to_fix or ""),
                    str(analysis.failure_mode or ""),
                    str(analysis.evidence or ""),
                ]
            )
        )
        scored_links: list[tuple[float, int, str, list[str]]] = []
        for theme in theme_rows:
            score = 0.0
            if analysis.block_id in theme["implicated_blocks"]:
                score += 2.4
            overlap = len(analysis_tokens.intersection(theme["tokens"]))
            if overlap > 0:
                score += min(1.8, overlap * 0.35)
            failure_mode = str(analysis.failure_mode or "").strip().lower()
            if failure_mode and failure_mode in str(theme["description"]).lower():
                score += 0.6
            if score >= 2.0:
                scored_links.append(
                    (
                        score,
                        int(theme["severity"]),
                        str(theme["theme_id"]),
                        list(theme["fix_patterns"]),
                    )
                )
        scored_links.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        linked_theme_ids = [item[2] for item in scored_links[:3]]
        fallback_issue_points = _dedupe_non_empty_strings(
            [analysis.what_to_fix, analysis.failure_mode, analysis.analysis],
            max_items=3,
        )
        linked_fix_patterns = _dedupe_non_empty_strings(
            list(analysis.recommended_fix_patterns or [])
            + [pattern for item in scored_links[:3] for pattern in item[3]],
            max_items=10,
        )
        analysis.linked_theme_ids = linked_theme_ids
        analysis.recommended_fix_patterns = linked_fix_patterns
        if not analysis.likely_issue_points:
            analysis.likely_issue_points = fallback_issue_points
        if analysis.confidence is not None and linked_theme_ids:
            adjusted_confidence = float(analysis.confidence) + (0.04 * len(linked_theme_ids))
            analysis.confidence = max(0.0, min(1.0, adjusted_confidence))


def _build_rca_prompt_v2(
    *,
    prompt: str,
    major_issues: str,
    valid_blocks: list[str],
    criteria_context_text: str,
    evidence_json: str,
    high_level_themes_context: str | None = None,
) -> str:
    themes_section = ""
    if high_level_themes_context:
        themes_section = f"High-level RCA themes to honor:\n{high_level_themes_context}\n\n"
    return (
        "You are an expert pipeline RCA analyst.\n"
        "Diagnose why the case failed and map findings to block-level fixes.\n"
        "You MUST only reference block IDs from the valid block list.\n"
        "Use evidence from tool execution, degraded markers, and step coverage.\n"
        "Do not overfit to this one case; avoid proposing case-specific facts or answer fragments.\n"
        "Each suggested fix must preserve output schema/contract safety.\n\n"
        f"{themes_section}"
        f"Valid blocks:\n{', '.join(valid_blocks)}\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Major issues from grading:\n{major_issues}\n\n"
        f"Evidence bundle:\n{evidence_json}\n\n"
        f"Block prompt criteria context:\n{criteria_context_text}\n\n"
        "Output requirements:\n"
        "- Provide concise analysis.\n"
        "- Set need_fix=true only when evidence supports it.\n"
        "- Include failure_mode, confidence (0..1), and concrete evidence references when possible.\n"
        "- Include likely_issue_points as concise actionable bullets.\n"
        "- If high-level themes are provided, populate linked_theme_ids with matching theme_id values.\n"
        "- Include reusable recommended_fix_patterns; do not include case-specific answers.\n"
        "- Keep recommendations generic and reusable."
    )


def _build_high_level_rca_prompt(
    *,
    failed_results: list[dict[str, Any]],
    valid_blocks: list[str],
) -> str:
    cohort: list[dict[str, Any]] = []
    for item in failed_results[:60]:
        case = item.get("case", {}) if isinstance(item, dict) else {}
        case_id = _case_identifier(case) if isinstance(case, dict) else "unknown_case"
        grade = item.get("grade", {}) if isinstance(item, dict) else {}
        case_stats = item.get("case_stats", {}) if isinstance(item, dict) else {}
        run_output = item.get("run_output", {}) if isinstance(item, dict) else {}
        cohort.append(
            {
                "case_id": case_id,
                "category": (case or {}).get("category"),
                "difficulty": (case or {}).get("difficulty"),
                "aggregate_score": float((grade or {}).get("aggregate_score", 0.0) or 0.0),
                "major_issues": str((grade or {}).get("major_issues", ""))[:600],
                "timed_out": bool((case_stats or {}).get("timed_out")),
                "degraded_mode_active": bool((case_stats or {}).get("degraded_mode_active")),
                "implicated_blocks": _implicated_blocks_for_case(
                    run_output if isinstance(run_output, dict) else {},
                    valid_blocks,
                )[:10],
                "tool_failure_signals": _prune_for_rca(
                    (run_output or {}).get("tool_failure_signals"),
                    max_depth=4,
                    max_items=8,
                    max_text_chars=220,
                ),
            }
        )
    return (
        "You are performing a HIGH-LEVEL RCA across multiple failed cases.\n"
        "Extract recurring, cross-case themes and prioritize reusable fixes.\n"
        "Do NOT propose case-specific answer hacks.\n"
        "Use only block IDs from the valid list.\n\n"
        f"Valid blocks:\n{', '.join(valid_blocks)}\n\n"
        f"Failed cohort summary:\n{_safe_json_dumps(cohort, max_chars=24000)}\n\n"
        "Return concise themes with evidence_case_ids, implicated_blocks, severity, confidence, and reusable fix patterns.\n"
        "Each theme_id should be stable (short snake_case) and map to concrete reusable remediation patterns."
    )


def run_high_level_rca(
    *,
    failed_results: list[dict[str, Any]],
    valid_blocks: list[str],
) -> dict[str, Any]:
    if not failed_results:
        return {"summary": "No RCA cohort selected", "themes": []}

    prompt = _build_high_level_rca_prompt(
        failed_results=failed_results,
        valid_blocks=valid_blocks,
    )
    report, used_model = _generate_with_oss_fallback(
        prompt=prompt,
        schema=HighLevelRCAReportSchema,
        temperature=0.2,
    )
    themes: list[dict[str, Any]] = []
    valid_block_set = set(valid_blocks)
    for theme in report.themes:
        implicated = [block_id for block_id in theme.implicated_blocks if block_id in valid_block_set]
        themes.append(
            {
                "theme_id": str(theme.theme_id),
                "description": str(theme.description),
                "evidence_case_ids": [str(case_id) for case_id in theme.evidence_case_ids[:24]],
                "implicated_blocks": implicated,
                "severity": int(theme.severity),
                "confidence": float(theme.confidence) if theme.confidence is not None else None,
                "recommended_fix_patterns": [str(item) for item in theme.recommended_fix_patterns[:12]],
            }
        )
    return {
        "summary": str(report.summary),
        "themes": themes,
        "model_used": used_model,
    }


def root_cause_analysis(
    prompt: str,
    run_output: dict[str, Any],
    major_issues: str,
    valid_blocks: list[str],
    block_details_map: dict[str, dict[str, Any]],
    criteria_context_text: str | None = None,
    case_context: dict[str, Any] | None = None,
    high_level_themes: list[dict[str, Any]] | None = None,
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

    evidence_bundle = _build_rca_evidence_bundle(run_output, case_context)
    evidence_json = _safe_json_dumps(evidence_bundle, max_chars=16000)
    high_level_context = None
    if high_level_themes:
        compact_themes = []
        for theme in high_level_themes[:10]:
            if not isinstance(theme, dict):
                continue
            compact_themes.append(
                {
                    "theme_id": theme.get("theme_id"),
                    "description": theme.get("description"),
                    "implicated_blocks": list(theme.get("implicated_blocks") or [])[:8],
                    "severity": theme.get("severity"),
                    "recommended_fix_patterns": list(theme.get("recommended_fix_patterns") or [])[:6],
                }
            )
        if compact_themes:
            high_level_context = _safe_json_dumps(compact_themes, max_chars=5000)
    rca_prompt = _build_rca_prompt_v2(
        prompt=prompt,
        major_issues=major_issues,
        valid_blocks=valid_blocks,
        criteria_context_text=criteria_context_text,
        evidence_json=evidence_json,
        high_level_themes_context=high_level_context,
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
            if not item.likely_issue_points:
                item.likely_issue_points = _dedupe_non_empty_strings(
                    [item.what_to_fix, item.failure_mode, item.analysis],
                    max_items=3,
                )
            if not item.recommended_fix_patterns:
                item.recommended_fix_patterns = _dedupe_non_empty_strings(
                    [item.what_to_fix],
                    max_items=4,
                )
            analyses.append(item)
    _link_analyses_to_high_level_themes(analyses, high_level_themes)
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
    if not isinstance(result, PromptCriteriaScoreSchema):
        logger.warning(
            "Unexpected scorer result type for block=%s: %s; applying conservative fallback score",
            block_id,
            type(result).__name__,
        )
        result = PromptCriteriaScoreSchema(
            generic_quality_score=1,
            criteria_alignment_score=1,
            anti_overfit_score=1,
            notes="[Graceful degradation] scorer returned non-schema output.",
        )
    return result, used_model


def _select_fallback_block_id(
    grouped: dict[str, list[BlockAnalysisSchema]],
    current_prompts: dict[str, str],
) -> str | None:
    ranked: list[tuple[int, float, str]] = []
    for block_id, analyses in grouped.items():
        if block_id not in current_prompts:
            continue
        if not analyses:
            continue
        issue_weight = sum(
            float(item.generic_issue_score) + float(item.criteria_misalignment_score)
            for item in analyses
        )
        ranked.append((len(analyses), issue_weight, block_id))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _deterministic_fallback_addendum(block_id: str) -> str:
    by_block: dict[str, str] = {
        "sub_plan_creation_block": (
            "Reliability addendum:\n"
            "- Return strict JSON with keys `sub_plan`, `steps`, and `tool_uses`.\n"
            "- `sub_plan` must be a string only, and `steps`/`tool_uses` must be arrays of strings."
        ),
        "self_critique_block": (
            "Reliability addendum:\n"
            "- Return strict JSON keys `given_item`, `general_critique`, and `list_of_issues`.\n"
            "- `list_of_issues` must contain plain strings only."
        ),
        "primary_tool_router_block": (
            "Reliability addendum:\n"
            "- Always return top-level `routes` as an array and `continuity` as a boolean.\n"
            "- Never return top-level `id`/`inputs` without wrapping them in `routes`."
        ),
        "secondary_tool_router_block": (
            "Reliability addendum:\n"
            "- Always return top-level `routes` as an array and `continuity` as a boolean.\n"
            "- Never return top-level `id`/`inputs` without wrapping them in `routes`."
        ),
        "python_code_execution_tool_block": (
            "Reliability addendum:\n"
            "- Preserve strict schema keys `code_to_run` and `packages_needed`.\n"
            "- Include direct repair behavior using `previous_error` and `previous_code`."
        ),
        "web_search_tool_block": (
            "Reliability addendum:\n"
            "- Cite only supplied reference IDs and remove unsupported claims.\n"
            "- If evidence is weak, state uncertainty plainly."
        ),
    }
    return by_block.get(
        block_id,
        (
            "Reliability addendum:\n"
            "- Keep output strictly aligned to schema keys and avoid unsupported assumptions."
        ),
    )


def _extract_route_ids(routing_payload: Any) -> set[str]:
    route_ids: set[str] = set()
    if not isinstance(routing_payload, dict):
        return route_ids
    routes = routing_payload.get("routes")
    if not isinstance(routes, list):
        return route_ids
    for route in routes:
        if not isinstance(route, dict):
            continue
        route_id = route.get("id")
        if route_id is None:
            continue
        route_ids.add(str(route_id))
    return route_ids


def _implicated_blocks_for_case(run_output: dict[str, Any], valid_blocks: list[str]) -> list[str]:
    valid_set = set(valid_blocks)
    implicated: set[str] = set()

    tool_metrics = _extract_case_tool_metrics(run_output)
    for key in ("tool_invocations", "tool_errors"):
        payload = tool_metrics.get(key, {})
        if not isinstance(payload, dict):
            continue
        for block_id, count in payload.items():
            if isinstance(count, (int, float)) and float(count) > 0:
                implicated.add(str(block_id))

    tool_context = run_output.get("tool_context")
    if isinstance(tool_context, dict):
        implicated.update(_extract_route_ids(tool_context.get("primary_routing")))
        implicated.update(_extract_route_ids(tool_context.get("primary_continuity_routing")))
        for subplan in tool_context.get("subplans", []):
            if not isinstance(subplan, dict):
                continue
            implicated.update(_extract_route_ids(subplan.get("routes")))
            implicated.update(_extract_route_ids(subplan.get("continuity_routes")))

    step_coverage = run_output.get("step_coverage")
    coverage_ratio = (
        float(step_coverage.get("coverage_ratio"))
        if isinstance(step_coverage, dict) and isinstance(step_coverage.get("coverage_ratio"), (int, float))
        else None
    )
    if coverage_ratio is not None and coverage_ratio < 1.0:
        implicated.add("synthesis_block")
    if run_output.get("degraded_mode_active"):
        implicated.add("synthesis_block")
    if run_output.get("plan_review"):
        implicated.update({"self_critique_block", "improvement_block"})
    if run_output.get("synthesis_review"):
        implicated.update({"self_critique_block", "improvement_block", "synthesis_block"})

    filtered = sorted(block_id for block_id in implicated if block_id in valid_set)
    if filtered:
        return filtered
    return list(valid_blocks)


def _criteria_context_for_blocks(
    block_ids: list[str],
    block_details_map: dict[str, dict[str, Any]],
) -> str:
    sections: list[str] = []
    for block_id in block_ids:
        details = block_details_map.get(block_id, {})
        criteria = details.get("prompt_creation_parameters", {})
        sections.append(
            f"Block: {block_id}\nPrompt criteria:\n{_safe_json_dumps(criteria, max_chars=900)}"
        )
    return "\n\n".join(sections)


def _select_rca_cases(
    results: list[dict[str, Any]],
    rca_case_budget: int,
    rca_threshold: float = DEFAULT_RCA_THRESHOLD,
    rca_fallback_fraction: float = DEFAULT_RCA_FALLBACK_FRACTION,
    case_history: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configured_budget = max(0, int(rca_case_budget))
    budget = configured_budget
    if budget == 0:
        return [], {
            "rule": "disabled",
            "rca_case_budget": 0,
            "configured_rca_case_budget": int(configured_budget),
            "threshold": float(rca_threshold),
            "fallback_fraction": float(rca_fallback_fraction),
            "total_cases": len(results),
            "selected_count": 0,
            "threshold_hits": 0,
            "fallback_count_before_budget": 0,
            "selected_case_ids": [],
        }

    ranked = sorted(
        [
            item
            for item in results
            if isinstance(item, dict) and isinstance(item.get("grade"), dict)
        ],
        key=lambda item: float(item.get("grade", {}).get("aggregate_score", 0.0) or 0.0),
    )
    threshold_matches = [
        item
        for item in ranked
        if float(item.get("grade", {}).get("aggregate_score", 0.0) or 0.0) < float(rca_threshold)
    ]

    if ENABLE_REPLAY_REBALANCE_V1 and ranked:
        dynamic_budget = min(
            6,
            max(
                3,
                int(math.ceil(0.35 * len(threshold_matches))),
                int(math.ceil(0.25 * len(ranked))),
            ),
        )
        budget = int(dynamic_budget)

    selection_rule = "threshold"
    selected_pool = threshold_matches
    fallback_count_before_budget = 0
    if not threshold_matches:
        selection_rule = "fallback_worst_fraction"
        bounded_fraction = max(0.05, min(1.0, float(rca_fallback_fraction)))
        fallback_count_before_budget = max(1, int(round(len(ranked) * bounded_fraction)))
        selected_pool = ranked[:fallback_count_before_budget]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _append_selected(item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        case_payload = item.get("case")
        case_id = _case_identifier(case_payload if isinstance(case_payload, dict) else {})
        if case_id in selected_ids:
            return
        selected.append(item)
        selected_ids.add(case_id)

    # Always include top-2 worst scoring items.
    for item in ranked[:2]:
        if len(selected) >= budget:
            break
        _append_selected(item)

    # Include one historical severe regression case when available.
    if len(selected) < budget and isinstance(case_history, dict):
        severe_candidates: list[tuple[float, dict[str, Any]]] = []
        for item in ranked:
            case_payload = item.get("case")
            if not isinstance(case_payload, dict):
                continue
            case_id = _case_identifier(case_payload)
            history = case_history.get(case_id) or {}
            last_delta = float(history.get("last_delta") or 0.0)
            timeout_count = int(history.get("timeout_count") or 0)
            if last_delta <= float(TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD) or timeout_count > 0:
                severe_candidates.append((_replay_priority_score(history), item))
        severe_candidates.sort(key=lambda pair: pair[0], reverse=True)
        for _, item in severe_candidates:
            if len(selected) >= budget:
                break
            _append_selected(item)
            break

    # Fill from threshold selection pool, then full ranking.
    for item in selected_pool:
        if len(selected) >= budget:
            break
        _append_selected(item)
    for item in ranked:
        if len(selected) >= budget:
            break
        _append_selected(item)

    selected_case_ids = [
        _case_identifier(item.get("case", {}))
        for item in selected
        if isinstance(item.get("case"), dict)
    ]
    diagnostics = {
        "rule": selection_rule,
        "threshold": float(rca_threshold),
        "fallback_fraction": float(rca_fallback_fraction),
        "rca_case_budget": int(budget),
        "configured_rca_case_budget": int(configured_budget),
        "total_cases": len(ranked),
        "threshold_hits": len(threshold_matches),
        "fallback_count_before_budget": fallback_count_before_budget,
        "selected_count": len(selected),
        "selected_case_ids": selected_case_ids,
        "dynamic_budget_enabled": bool(ENABLE_REPLAY_REBALANCE_V1),
    }
    return selected, diagnostics


def _rank_blocks_by_rca_impact(
    grouped: dict[str, list[BlockAnalysisSchema]],
    *,
    block_impact_history: dict[str, dict[str, Any]] | None = None,
    enable_block_impact_ranker: bool = ENABLE_BLOCK_IMPACT_RANKER,
) -> list[tuple[float, str]]:
    ranked: list[tuple[float, str]] = []
    history = block_impact_history or {}
    for block_id, analyses in grouped.items():
        if not analyses:
            continue
        rca_impact = sum(
            float(analysis.generic_issue_score) + float(analysis.criteria_misalignment_score)
            for analysis in analyses
        )
        linkage_bonus = sum(
            (0.45 * len(set(analysis.linked_theme_ids or [])))
            + (0.12 * len(analysis.recommended_fix_patterns or []))
            + (0.08 * len(analysis.likely_issue_points or []))
            for analysis in analyses
        )
        impact = float(rca_impact) + float(linkage_bonus)
        if enable_block_impact_ranker:
            block_hist = history.get(str(block_id)) or {}
            avg_mean_delta = float(block_hist.get("avg_mean_delta", 0.0) or 0.0)
            acceptance_rate = float(block_hist.get("acceptance_rate", 0.0) or 0.0)
            # Historical end-to-end signal nudges ranking toward blocks that have helped before.
            impact += (avg_mean_delta * 3.0) + acceptance_rate
        ranked.append((impact, block_id))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def _update_block_impact_history(
    *,
    block_impact_history: dict[str, dict[str, Any]],
    mutated_block_ids: list[str],
    winner: str,
    train_delta_stats: dict[str, Any],
) -> None:
    if not mutated_block_ids:
        return
    mean_delta = float(train_delta_stats.get("mean_delta", 0.0) or 0.0)
    p10_delta = float(_percentile(list(train_delta_stats.get("deltas") or []), 10) or mean_delta)
    worst_case_delta = float(min(list(train_delta_stats.get("deltas") or [mean_delta])))
    accepted = str(winner or "") == "candidate"

    for block_id in mutated_block_ids:
        if not block_id:
            continue
        entry = block_impact_history.setdefault(
            str(block_id),
            {
                "times_mutated": 0,
                "times_accepted": 0,
                "sum_mean_delta": 0.0,
                "sum_p10_delta": 0.0,
                "worst_observed_delta": worst_case_delta,
                "avg_mean_delta": 0.0,
                "avg_p10_delta": 0.0,
                "acceptance_rate": 0.0,
            },
        )
        entry["times_mutated"] = int(entry.get("times_mutated", 0) or 0) + 1
        if accepted:
            entry["times_accepted"] = int(entry.get("times_accepted", 0) or 0) + 1
        entry["sum_mean_delta"] = float(entry.get("sum_mean_delta", 0.0) or 0.0) + mean_delta
        entry["sum_p10_delta"] = float(entry.get("sum_p10_delta", 0.0) or 0.0) + p10_delta
        previous_worst = float(entry.get("worst_observed_delta", worst_case_delta) or worst_case_delta)
        entry["worst_observed_delta"] = min(previous_worst, worst_case_delta)
        mutated = max(1, int(entry.get("times_mutated", 1) or 1))
        accepted_count = int(entry.get("times_accepted", 0) or 0)
        entry["avg_mean_delta"] = float(entry["sum_mean_delta"]) / float(mutated)
        entry["avg_p10_delta"] = float(entry["sum_p10_delta"]) / float(mutated)
        entry["acceptance_rate"] = float(accepted_count) / float(mutated)


RECOVERABLE_MUTATION_FAILURE_CODES: set[str] = {
    "missing_required_placeholders",
    "prompt_contract_violation",
    "candidate_same_as_baseline",
}


def _dedupe_block_analyses(analyses: list[BlockAnalysisSchema]) -> list[BlockAnalysisSchema]:
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
    return unique_analyses


MUTATION_STYLE_LIBRARY: dict[str, dict[str, str]] = {
    "failure_targeted_patch": {
        "title": "Failure-targeted patch",
        "change_budget": "Change only the minimum wording needed for the top RCA failure.",
        "instructions": (
            "Focus tightly on the highest-confidence failure mode and patch it with minimal drift."
        ),
    },
    "structure_tightening": {
        "title": "Structure tightening",
        "change_budget": "You may rewrite up to 25% of the prompt while preserving scaffolding intent.",
        "instructions": (
            "Improve hierarchy, ordering, and instruction boundaries so behavior is easier to follow."
        ),
    },
    "contract_hardening": {
        "title": "Contract hardening",
        "change_budget": "Prioritize contract-safe edits even if wording changes are broader.",
        "instructions": (
            "Strengthen placeholder handling, schema guarantees, and strict format constraints."
        ),
    },
    "reliability_guardrails": {
        "title": "Reliability guardrails",
        "change_budget": "Add robust fallback and error-handling directives when failures suggest instability.",
        "instructions": (
            "Reduce timeout/degraded/tool-failure risk with concise guardrails that remain generic."
        ),
    },
}
MUTATION_STYLE_DEFAULT_ORDER = [
    "failure_targeted_patch",
    "structure_tightening",
    "contract_hardening",
    "reliability_guardrails",
]


def _collect_analysis_linkage_summary(analyses: list[BlockAnalysisSchema]) -> dict[str, Any]:
    linked_theme_ids = _dedupe_non_empty_strings(
        [theme_id for analysis in analyses for theme_id in (analysis.linked_theme_ids or [])],
        max_items=24,
    )
    recommended_fix_patterns = _dedupe_non_empty_strings(
        [pattern for analysis in analyses for pattern in (analysis.recommended_fix_patterns or [])],
        max_items=16,
    )
    likely_issue_points = _dedupe_non_empty_strings(
        [point for analysis in analyses for point in (analysis.likely_issue_points or [])],
        max_items=16,
    )
    dominant_failure_modes = _dedupe_non_empty_strings(
        [analysis.failure_mode for analysis in analyses if analysis.failure_mode],
        max_items=8,
    )
    return {
        "linked_theme_ids": linked_theme_ids,
        "recommended_fix_patterns": recommended_fix_patterns,
        "likely_issue_points": likely_issue_points,
        "dominant_failure_modes": dominant_failure_modes,
    }


def _rank_mutation_styles_for_block(analyses: list[BlockAnalysisSchema]) -> list[str]:
    text = " ".join(
        [
            str(analysis.analysis or "")
            + " "
            + str(analysis.what_to_fix or "")
            + " "
            + str(analysis.failure_mode or "")
            for analysis in analyses
        ]
    ).lower()
    styles: list[str] = ["failure_targeted_patch"]

    if re.search(r"\b(json|schema|contract|placeholder|format|parser|strict)\b", text):
        styles.append("contract_hardening")
    if re.search(r"\b(timeout|degraded|tool|failure|retry|latency|error|fallback)\b", text):
        styles.append("reliability_guardrails")
    if re.search(r"\b(ambiguous|clarity|confusing|ordering|structure|hierarchy)\b", text):
        styles.append("structure_tightening")

    for style in MUTATION_STYLE_DEFAULT_ORDER:
        if style not in styles:
            styles.append(style)
    return styles


def _build_mutation_style_plan(
    analyses: list[BlockAnalysisSchema],
    *,
    tournament_size: int,
) -> list[str]:
    ranked_styles = _rank_mutation_styles_for_block(analyses)
    target_size = max(1, int(tournament_size))
    desired_unique = max(1, min(target_size, int(DEFAULT_MUTATION_DIVERSITY_TARGET)))
    plan: list[str] = []
    for style in ranked_styles:
        plan.append(style)
        if len(plan) >= target_size:
            break
    if len(set(plan)) < desired_unique:
        for style in MUTATION_STYLE_DEFAULT_ORDER:
            if style in plan:
                continue
            plan.append(style)
            if len(plan) >= target_size:
                break
            if len(set(plan)) >= desired_unique:
                break
    while len(plan) < target_size:
        plan.append(ranked_styles[len(plan) % len(ranked_styles)])
    return plan[:target_size]


def _mutation_style_for_retry(base_style: str, failure_code: str | None) -> str:
    reason = str(failure_code or "").strip()
    if reason in {"missing_required_placeholders", "prompt_contract_violation"}:
        return "contract_hardening"
    if reason == "candidate_same_as_baseline" and base_style == "failure_targeted_patch":
        return "structure_tightening"
    return base_style


def _mutation_temperature_for_attempt(attempt_index: int, mutation_style: str | None = None) -> float:
    index = max(0, min(int(attempt_index), len(MUTATION_RETRY_TEMPERATURE_SCHEDULE) - 1))
    base_temperature = float(MUTATION_RETRY_TEMPERATURE_SCHEDULE[index])
    style = str(mutation_style or "").strip()
    if style == "structure_tightening":
        return min(0.60, base_temperature + 0.05)
    if style == "contract_hardening":
        return max(0.10, base_temperature - 0.05)
    return base_temperature


def _build_mutation_prompt(
    *,
    block_id: str,
    current_prompt: str,
    criteria: dict[str, Any],
    analyses: list[BlockAnalysisSchema],
    mutation_style: str,
    style_plan: list[str] | None = None,
    required_placeholders: list[str],
    previous_candidate: str | None = None,
    failure_code: str | None = None,
    failure_details: dict[str, Any] | None = None,
) -> str:
    style_id = str(mutation_style or "failure_targeted_patch")
    style_config = MUTATION_STYLE_LIBRARY.get(style_id, MUTATION_STYLE_LIBRARY["failure_targeted_patch"])
    linkage_summary = _collect_analysis_linkage_summary(analyses)
    analyses_text = "\n".join(
        (
            f"- generic_issue_score={analysis.generic_issue_score}, "
            f"criteria_misalignment_score={analysis.criteria_misalignment_score}\n"
            f"  analysis={analysis.analysis}\n"
            f"  what_to_fix={analysis.what_to_fix}\n"
            f"  failure_mode={analysis.failure_mode or 'unknown'}\n"
            f"  confidence={analysis.confidence if analysis.confidence is not None else 'n/a'}\n"
            f"  evidence={analysis.evidence or 'n/a'}\n"
            f"  linked_theme_ids={analysis.linked_theme_ids or []}\n"
            f"  recommended_fix_patterns={analysis.recommended_fix_patterns or []}\n"
            f"  likely_issue_points={analysis.likely_issue_points or []}"
        )
        for analysis in analyses
    )

    # --- Failure-focused context: extract concrete failing outputs ----------
    failure_examples_text = ""
    failure_examples = []
    for analysis in analyses:
        evidence = getattr(analysis, "evidence", None) or ""
        what_to_fix = getattr(analysis, "what_to_fix", None) or ""
        if evidence and len(evidence) > 20:
            failure_examples.append(
                f"  * Evidence: {evidence[:400]}\n"
                f"    Fix needed: {what_to_fix[:200]}"
            )
    if failure_examples:
        failure_examples_text = (
            "\nConcrete failure examples from A/B test cases (use these to guide your fix):\n"
            + "\n".join(failure_examples[:5])
            + "\n"
        )

    contract_checklist = (
        "Contract checklist (must pass all):\n"
        f"- Preserve required placeholders exactly: {required_placeholders}\n"
        "- Preserve schema/output-contract markers and strict JSON requirements when present.\n"
        "- Do not remove essential section scaffolding that controls behavior.\n"
        "- Do not introduce conflicting 'plain text only' instructions.\n"
        "- Keep the revised prompt generic; never include case-specific facts or answer fragments.\n"
    )

    retry_instructions = ""
    if previous_candidate is not None:
        retry_instructions = (
            "Repair mode:\n"
            f"- Prior candidate failed with code: {failure_code or 'unknown'}.\n"
            f"- Failure diagnostics: {_safe_json_dumps(failure_details or {}, max_chars=1400)}\n"
            "Fix only what caused failure while preserving improvements from RCA.\n"
            f"Prior candidate prompt:\n{previous_candidate}\n\n"
        )

    style_plan_text = ", ".join(style_plan or [])
    linked_theme_text = _safe_json_dumps(
        {
            "linked_theme_ids": linkage_summary.get("linked_theme_ids", []),
            "recommended_fix_patterns": linkage_summary.get("recommended_fix_patterns", []),
            "likely_issue_points": linkage_summary.get("likely_issue_points", []),
            "dominant_failure_modes": linkage_summary.get("dominant_failure_modes", []),
        },
        max_chars=1800,
    )

    return (
        "You are an expert prompt engineer. Improve this block prompt while preserving strong baseline behavior.\n"
        f"Mutation style: {style_id} ({style_config.get('title')})\n"
        f"Style intent: {style_config.get('instructions')}\n"
        f"Style change budget: {style_config.get('change_budget')}\n"
        f"Tournament style plan: {style_plan_text}\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Target the SPECIFIC failure mode identified in RCA, not general improvements.\n"
        "- Preserve baseline section scaffolding and placeholders unless explicitly required.\n"
        "- Do not insert specific training-case facts, numbers, or direct answer fragments.\n\n"
        f"Block ID: {block_id}\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"Linked RCA themes and fix patterns:\n{linked_theme_text}\n\n"
        f"RCA analyses:\n{analyses_text}\n\n"
        f"{failure_examples_text}"
        f"Prompt creation parameters:\n{_safe_json_dumps(criteria, max_chars=2500)}\n\n"
        f"{contract_checklist}\n"
        f"{retry_instructions}"
        "Editing protocol:\n"
        "- Keep existing section headings and ordering where possible.\n"
        "- Modify only the text needed to address the top-priority RCA findings for this style.\n"
        "- Keep wording concise and generalizable.\n"
        "- Preserve strict JSON/output contract language for schema-based blocks.\n"
        "Return a concise `plan_for_improvement` (1-2 sentences max describing exactly what you changed) "
        "and the revised prompt text in `prompt`."
    )


def _is_recoverable_mutation_failure(reason: str) -> bool:
    return str(reason or "").strip() in RECOVERABLE_MUTATION_FAILURE_CODES


def _validate_mutation_candidate(
    *,
    block_id: str,
    current_prompt: str,
    candidate_prompt: str,
    required_placeholders: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    if candidate_prompt.strip() == current_prompt.strip():
        return {
            "valid": False,
            "decision_reason": "candidate_same_as_baseline",
            "missing_placeholders": [],
            "contract_issues": [],
            "contract_hard_issues": [],
            "contract_soft_issues": [],
            "contract_auto_repair_applied": [],
            "candidate_prompt": candidate_prompt,
        }

    missing = [
        token for token in required_placeholders if token not in _placeholder_tokens(candidate_prompt)
    ]
    if missing:
        return {
            "valid": False,
            "decision_reason": "missing_required_placeholders",
            "missing_placeholders": missing,
            "contract_issues": [],
            "contract_hard_issues": [],
            "contract_soft_issues": [],
            "contract_auto_repair_applied": [],
            "candidate_prompt": candidate_prompt,
        }

    contract_issues = _validate_prompt_contract(block_id, candidate_prompt, details)
    contract_hard_issues, contract_soft_issues = _split_contract_issues(contract_issues)
    auto_repair_applied: list[str] = []
    if contract_soft_issues and not contract_hard_issues:
        repaired_prompt, auto_repair_applied = _auto_repair_contract_issues(
            block_id,
            candidate_prompt,
            contract_soft_issues,
        )
        if repaired_prompt != candidate_prompt:
            candidate_prompt = repaired_prompt
            contract_issues = _validate_prompt_contract(block_id, candidate_prompt, details)
            contract_hard_issues, contract_soft_issues = _split_contract_issues(contract_issues)

    if contract_hard_issues:
        return {
            "valid": False,
            "decision_reason": "prompt_contract_violation",
            "missing_placeholders": [],
            "contract_issues": contract_issues,
            "contract_hard_issues": contract_hard_issues,
            "contract_soft_issues": contract_soft_issues,
            "contract_auto_repair_applied": auto_repair_applied,
            "candidate_prompt": candidate_prompt,
        }

    return {
        "valid": True,
        "decision_reason": "candidate_structurally_valid",
        "missing_placeholders": [],
        "contract_issues": contract_issues,
        "contract_hard_issues": contract_hard_issues,
        "contract_soft_issues": contract_soft_issues,
        "contract_auto_repair_applied": auto_repair_applied,
        "candidate_prompt": candidate_prompt,
    }


def mutate_block_with_retries(
    *,
    block_id: str,
    current_prompt: str,
    analyses: list[BlockAnalysisSchema],
    details: dict[str, Any],
    mutation_style: str,
    style_plan: list[str] | None = None,
    mutation_retry_enabled: bool,
    mutation_max_retries: int,
    progress_tracker: TrainingProgressTracker | None = None,
    epoch_current: int | None = None,
) -> dict[str, Any]:
    required_placeholders = sorted(_placeholder_tokens(current_prompt))
    criteria = details.get("prompt_creation_parameters", {})
    max_retries = max(0, int(mutation_max_retries))
    total_attempts = 1 + max_retries if mutation_retry_enabled else 1

    attempt_history: list[dict[str, Any]] = []
    previous_candidate: str | None = None
    previous_failure: dict[str, Any] | None = None
    base_style = str(mutation_style or "failure_targeted_patch")
    mutation_style_used = base_style
    retry_budget_by_reason = {
        "prompt_contract_violation": 1,
        "missing_required_placeholders": 1,
        "candidate_same_as_baseline": 2,
    }
    retry_count_by_reason: dict[str, int] = {}

    for attempt_idx in range(total_attempts):
        is_retry = attempt_idx > 0
        mutation_style_used = (
            _mutation_style_for_retry(base_style, (previous_failure or {}).get("decision_reason"))
            if is_retry
            else base_style
        )
        temperature = _mutation_temperature_for_attempt(attempt_idx, mutation_style=mutation_style_used)
        mutation_prompt = _build_mutation_prompt(
            block_id=block_id,
            current_prompt=current_prompt,
            criteria=criteria,
            analyses=analyses,
            mutation_style=mutation_style_used,
            style_plan=style_plan,
            required_placeholders=required_placeholders,
            previous_candidate=previous_candidate if is_retry else None,
            failure_code=(previous_failure or {}).get("decision_reason") if is_retry else None,
            failure_details=previous_failure if is_retry else None,
        )

        mutation, used_model = _generate_with_oss_fallback(
            prompt=mutation_prompt,
            schema=PromptMutationSchema,
            temperature=temperature,
        )
        candidate_prompt = str(mutation.prompt or "")
        validation = _validate_mutation_candidate(
            block_id=block_id,
            current_prompt=current_prompt,
            candidate_prompt=candidate_prompt,
            required_placeholders=required_placeholders,
            details=details,
        )

        attempt_record = {
            "attempt_index": attempt_idx,
            "temperature": temperature,
            "mutation_model": used_model,
            "mutation_style": mutation_style_used,
            "decision_reason": validation.get("decision_reason"),
            "missing_placeholders": list(validation.get("missing_placeholders") or []),
            "contract_hard_issues": list(validation.get("contract_hard_issues") or []),
            "contract_soft_issues": list(validation.get("contract_soft_issues") or []),
            "valid": bool(validation.get("valid")),
        }
        attempt_history.append(attempt_record)

        if validation.get("valid"):
            return {
                "valid": True,
                "candidate_prompt": validation.get("candidate_prompt", candidate_prompt),
                "mutation_model": used_model,
                "mutation_style": mutation_style_used,
                "required_placeholders": required_placeholders,
                "attempt_history": attempt_history,
                "style_plan": list(style_plan or []),
                **validation,
            }

        previous_candidate = candidate_prompt
        previous_failure = {
            "decision_reason": validation.get("decision_reason"),
            "missing_placeholders": list(validation.get("missing_placeholders") or []),
            "contract_hard_issues": list(validation.get("contract_hard_issues") or []),
            "contract_soft_issues": list(validation.get("contract_soft_issues") or []),
        }

        reason = str(validation.get("decision_reason") or "")
        has_next_attempt = attempt_idx < (total_attempts - 1)
        if has_next_attempt and _is_recoverable_mutation_failure(reason):
            retry_count_by_reason[reason] = retry_count_by_reason.get(reason, 0) + 1
            max_reason_retries = retry_budget_by_reason.get(reason)
            if max_reason_retries is not None and retry_count_by_reason[reason] > int(max_reason_retries):
                break
            if progress_tracker:
                progress_tracker.emit(
                    event_type="mutation_retry_attempt",
                    phase="prompt_improvement",
                    step="repair_mutation",
                    message=f"Retrying mutation for block {block_id} (attempt {attempt_idx + 2}/{total_attempts})",
                    epoch_current=epoch_current,
                    payload={
                        "block_id": block_id,
                        "attempt_index": attempt_idx + 1,
                        "total_attempts": total_attempts,
                        "failure_code": reason,
                        "retry_temperature": _mutation_temperature_for_attempt(
                            attempt_idx + 1,
                            mutation_style=_mutation_style_for_retry(base_style, reason),
                        ),
                        "mutation_style": mutation_style_used,
                    },
                )
            continue
        break

    final_reason = str((previous_failure or {}).get("decision_reason") or "mutation_retry_exhausted")
    if progress_tracker:
        progress_tracker.emit(
            event_type="mutation_retry_exhausted",
            phase="prompt_improvement",
            step="repair_mutation_exhausted",
            message=f"Mutation retries exhausted for block {block_id}",
            epoch_current=epoch_current,
            payload={
                "block_id": block_id,
                "failure_code": final_reason,
                "attempts": attempt_history,
            },
        )

    return {
        "valid": False,
        "candidate_prompt": previous_candidate or current_prompt,
        "mutation_model": attempt_history[-1]["mutation_model"] if attempt_history else None,
        "mutation_style": mutation_style_used,
        "required_placeholders": required_placeholders,
        "attempt_history": attempt_history,
        "style_plan": list(style_plan or []),
        "decision_reason": final_reason,
        "missing_placeholders": list((previous_failure or {}).get("missing_placeholders") or []),
        "contract_issues": list((previous_failure or {}).get("contract_hard_issues") or []),
        "contract_hard_issues": list((previous_failure or {}).get("contract_hard_issues") or []),
        "contract_soft_issues": list((previous_failure or {}).get("contract_soft_issues") or []),
        "contract_auto_repair_applied": [],
    }


def prompt_improvements(
    current_prompts: dict[str, str],
    block_analyses: list[BlockAnalysisSchema],
    block_details_map: dict[str, dict[str, Any]],
    mutation_block_budget: int = 1,
    mutation_accept_threshold: int = MUTATION_ACCEPTANCE_TOTAL_MIN,
    mutation_retry_enabled: bool = True,
    mutation_max_retries: int = DEFAULT_MUTATION_MAX_RETRIES,
    mutation_tournament_size: int = DEFAULT_MUTATION_TOURNAMENT_SIZE,
    block_impact_history: dict[str, dict[str, Any]] | None = None,
    progress_tracker: TrainingProgressTracker | None = None,
    epoch_current: int | None = None,
    target_block_ids_override: list[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    grouped: dict[str, list[BlockAnalysisSchema]] = defaultdict(list)
    for analysis in block_analyses:
        if analysis.need_fix:
            grouped[analysis.block_id].append(analysis)

    improvements: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], tuple[PromptCriteriaScoreSchema, str]] = {}
    ranked_blocks = _rank_blocks_by_rca_impact(
        grouped,
        block_impact_history=block_impact_history,
        enable_block_impact_ranker=ENABLE_BLOCK_IMPACT_RANKER,
    )
    target_block_ids = [block_id for _, block_id in ranked_blocks]
    if isinstance(target_block_ids_override, list) and target_block_ids_override:
        override_set = {str(item) for item in target_block_ids_override if str(item).strip()}
        target_block_ids = [block_id for block_id in target_block_ids if block_id in override_set]

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

    for block_id in target_block_ids:
        analyses = grouped.get(block_id, [])
        if block_id not in current_prompts:
            continue

        current_prompt = current_prompts[block_id]
        details = block_details_map.get(block_id, {})
        unique_analyses = _dedupe_block_analyses(analyses)
        analysis_linkage = _collect_analysis_linkage_summary(unique_analyses)

        # -- Tournament: generate N candidates, score each, pick best --------
        tournament_size = int(max(1, mutation_tournament_size))
        effective_tournament_size = int(max(1, min(tournament_size, 3))) if ENABLE_MUTATION_BUDGET_OPT_V1 else tournament_size
        style_plan = _build_mutation_style_plan(unique_analyses, tournament_size=effective_tournament_size)
        candidates: list[dict[str, Any]] = []  # [{mutation_result, candidate_prompt, ...}]
        all_tournament_attempts: list[dict[str, Any]] = []
        invalid_reason_counts: dict[str, int] = {}
        invalid_reasons_by_style: dict[str, dict[str, int]] = {}
        seen_candidate_hashes: set[str] = set()

        for t_idx in range(effective_tournament_size):
            if ENABLE_MUTATION_BUDGET_OPT_V1 and t_idx >= 1:
                best_so_far = max(
                    [float(item.get("candidate_total", 0.0) or 0.0) for item in candidates] or [0.0]
                )
                if t_idx == 1 and best_so_far >= 26.0:
                    break
                if t_idx >= 2 and best_so_far >= 25.5:
                    break
            style_for_slot = style_plan[t_idx % len(style_plan)]
            try:
                mutation_result = mutate_block_with_retries(
                    block_id=block_id,
                    current_prompt=current_prompt,
                    analyses=unique_analyses,
                    details=details,
                    mutation_style=style_for_slot,
                    style_plan=style_plan,
                    mutation_retry_enabled=bool(mutation_retry_enabled),
                    mutation_max_retries=int(max(0, mutation_max_retries)),
                    progress_tracker=progress_tracker,
                    epoch_current=epoch_current,
                )
            except Exception as exc:
                logger.warning(
                    "Tournament[%d/%d] mutation failed for block=%s: %s",
                    t_idx + 1,
                    tournament_size,
                    block_id,
                    exc,
                )
                all_tournament_attempts.append(
                    {
                        "tournament_idx": t_idx,
                        "mutation_style": style_for_slot,
                        "error": str(exc),
                    }
                )
                continue

            cand_prompt = str(mutation_result.get("candidate_prompt", current_prompt))
            attempt_history = list(mutation_result.get("attempt_history") or [])
            all_tournament_attempts.extend(attempt_history)
            mutation_style_used = str(mutation_result.get("mutation_style") or style_for_slot)
            candidate_hash = hashlib.sha1(cand_prompt.encode("utf-8", errors="ignore")).hexdigest()
            if candidate_hash in seen_candidate_hashes:
                invalid_reason = "duplicate_candidate_hash"
                invalid_reason_counts[invalid_reason] = invalid_reason_counts.get(invalid_reason, 0) + 1
                by_style = invalid_reasons_by_style.setdefault(mutation_style_used, {})
                by_style[invalid_reason] = by_style.get(invalid_reason, 0) + 1
                continue
            seen_candidate_hashes.add(candidate_hash)

            if not bool(mutation_result.get("valid")):
                invalid_reason = str(mutation_result.get("decision_reason") or "mutation_retry_exhausted")
                invalid_reason_counts[invalid_reason] = invalid_reason_counts.get(invalid_reason, 0) + 1
                by_style = invalid_reasons_by_style.setdefault(mutation_style_used, {})
                by_style[invalid_reason] = by_style.get(invalid_reason, 0) + 1
                logger.debug("Tournament[%d/%d] invalid candidate for block=%s", t_idx + 1, tournament_size, block_id)
                continue

            # Score the candidate
            try:
                baseline_score, baseline_score_model = _score_cached(block_id, current_prompt, details)
                candidate_score, candidate_score_model = _score_cached(block_id, cand_prompt, details)
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
            except Exception as exc:
                logger.warning("Tournament[%d/%d] scoring failed for block=%s: %s", t_idx + 1, tournament_size, block_id, exc)
                continue

            delta_total = candidate_total - baseline_total
            candidates.append({
                "candidate_prompt": cand_prompt,
                "mutation_result": mutation_result,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "baseline_score_model": baseline_score_model,
                "candidate_score_model": candidate_score_model,
                "baseline_total": baseline_total,
                "candidate_total": candidate_total,
                "delta_total": delta_total,
                "tournament_idx": t_idx,
                "mutation_style": mutation_style_used,
            })

        # -- Pick the best candidate from the tournament ---------------------
        if not candidates:
            mapped_reason = "rejected_retry_exhausted"
            if invalid_reason_counts:
                top_reason = sorted(
                    invalid_reason_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[0][0]
                if top_reason == "missing_required_placeholders":
                    mapped_reason = "rejected_placeholder"
                elif top_reason == "prompt_contract_violation":
                    mapped_reason = "rejected_contract"
            improvements[block_id] = current_prompt
            diagnostics.append({
                "block_id": block_id,
                "analysis_count": len(unique_analyses),
                "mutation_model": None,
                "changed": False,
                "accepted": False,
                "decision_reason": mapped_reason,
                "legacy_decision_reason": "tournament_no_valid_candidates",
                "tournament_size": effective_tournament_size,
                "tournament_size_requested": tournament_size,
                "tournament_budget_mode": "progressive_v1" if ENABLE_MUTATION_BUDGET_OPT_V1 else "fixed",
                "tournament_scored": 0,
                "required_placeholders_count": len(_placeholder_tokens(current_prompt)),
                "missing_placeholders": [],
                "baseline_total": None,
                "candidate_total": None,
                "delta_total": 0.0,
                "baseline_scores": {},
                "candidate_scores": {},
                "score_models": {},
                "candidate_prompt_preview": "",
                "mutation_attempts": all_tournament_attempts,
                "invalid_reason_counts": invalid_reason_counts,
                "invalid_reasons_by_style": invalid_reasons_by_style,
                "mutation_style_plan": style_plan,
                "mutation_styles_attempted": sorted(
                    {
                        str(item.get("mutation_style"))
                        for item in all_tournament_attempts
                        if isinstance(item, dict) and str(item.get("mutation_style") or "").strip()
                    }
                ),
                "analysis_linkage": analysis_linkage,
                "mutation_accept_threshold": int(mutation_accept_threshold),
            })
            continue

        # Sort by delta_total descending, pick best
        candidates.sort(
            key=lambda c: (
                float(c["candidate_total"]),
                float(c["delta_total"]),
                -int(c["tournament_idx"]),
            ),
            reverse=True,
        )
        best = candidates[0]

        candidate_prompt = best["candidate_prompt"]
        mutation_result = best["mutation_result"]
        used_model = mutation_result.get("mutation_model")
        selected_mutation_style = str(best.get("mutation_style") or mutation_result.get("mutation_style") or "")
        contract_issues = list(mutation_result.get("contract_issues") or [])
        contract_hard_issues = list(mutation_result.get("contract_hard_issues") or [])
        contract_soft_issues = list(mutation_result.get("contract_soft_issues") or [])
        auto_repair_applied = list(mutation_result.get("contract_auto_repair_applied") or [])
        required_placeholders = list(mutation_result.get("required_placeholders") or [])

        delta_total = best["delta_total"]
        candidate_total = float(best["candidate_total"])
        accepted = candidate_total >= float(mutation_accept_threshold)
        if accepted:
            improvements[block_id] = candidate_prompt
            decision_reason = "accepted_absolute_threshold"
        else:
            improvements[block_id] = current_prompt
            decision_reason = "rejected_below_absolute_threshold"

        diagnostics.append({
            "block_id": block_id,
            "analysis_count": len(unique_analyses),
            "mutation_model": used_model,
            "changed": True,
            "accepted": accepted,
            "decision_reason": decision_reason,
            "selected_mutation_style": selected_mutation_style,
            "mutation_style_plan": style_plan,
            "mutation_styles_attempted": sorted(
                {
                    str(item.get("mutation_style"))
                    for item in all_tournament_attempts
                    if isinstance(item, dict) and str(item.get("mutation_style") or "").strip()
                }
            ),
            "tournament_size": effective_tournament_size,
            "tournament_size_requested": tournament_size,
            "tournament_budget_mode": "progressive_v1" if ENABLE_MUTATION_BUDGET_OPT_V1 else "fixed",
            "tournament_scored": len(candidates),
            "tournament_deltas": [round(c["delta_total"], 3) for c in candidates],
            "tournament_styles": [str(c.get("mutation_style") or "") for c in candidates],
            "contract_issues": contract_issues,
            "contract_hard_issues": contract_hard_issues,
            "contract_soft_issues": contract_soft_issues,
            "contract_auto_repair_applied": auto_repair_applied,
            "required_placeholders_count": len(required_placeholders),
            "missing_placeholders": [],
            "baseline_total": best["baseline_total"],
            "candidate_total": candidate_total,
            "delta_total": delta_total,
            "baseline_scores": best["baseline_score"].model_dump(),
            "candidate_scores": best["candidate_score"].model_dump(),
            "score_models": {
                "baseline": best["baseline_score_model"],
                "candidate": best["candidate_score_model"],
            },
            "candidate_prompt_preview": candidate_prompt[:240],
            "mutation_attempts": all_tournament_attempts,
            "invalid_reason_counts": invalid_reason_counts,
            "invalid_reasons_by_style": invalid_reasons_by_style,
            "analysis_linkage": analysis_linkage,
            "mutation_accept_threshold": int(mutation_accept_threshold),
            "mutation_block_budget_deprecated": int(max(0, mutation_block_budget)),
        })

    return improvements, diagnostics


def _summarize_mutation_style_diagnostics(prompt_scoring: list[dict[str, Any]]) -> dict[str, Any]:
    style_attempt_counts: dict[str, int] = {}
    selected_style_counts: dict[str, int] = {}
    blocks_with_styles = 0
    for item in prompt_scoring:
        if not isinstance(item, dict):
            continue
        attempted = item.get("mutation_styles_attempted") or []
        if isinstance(attempted, list) and attempted:
            blocks_with_styles += 1
            for style in attempted:
                style_key = str(style).strip()
                if not style_key:
                    continue
                style_attempt_counts[style_key] = style_attempt_counts.get(style_key, 0) + 1
        selected_style = str(item.get("selected_mutation_style") or "").strip()
        if selected_style:
            selected_style_counts[selected_style] = selected_style_counts.get(selected_style, 0) + 1
    return {
        "blocks_with_styles": blocks_with_styles,
        "style_attempt_counts": style_attempt_counts,
        "selected_style_counts": selected_style_counts,
        "style_catalog": list(MUTATION_STYLE_DEFAULT_ORDER),
    }


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
        "overfit_risk_score": int(check.overfit_risk_score),
        "suspicious_phrases": merged_suspicious,
        "rationale": check.rationale,
        "heuristic_suspicious_count": len(suspicious),
        "model_used": model_used,
    }


def evaluate_suite(
    prompt_suite: dict[str, str],
    selected_cases: list[dict[str, Any]],
    thinking_level: Literal["low", "med-synth", "med-plan", "high"],
    phase: str = "evaluation",
    epoch_current: int | None = None,
    progress_tracker: TrainingProgressTracker | None = None,
    reference_scores: list[float] | None = None,
    early_stop_min_cases: int = EARLY_STOP_MIN_CASES,
    early_stop_mean_delta_threshold: float = EARLY_STOP_MEAN_DELTA_THRESHOLD,
    eval_mode: str = DEFAULT_EVAL_MODE,
    eval_min_pairs: int = DEFAULT_TRAIN_EVAL_MIN_PAIRS,
    eval_max_pairs: int = DEFAULT_TRAIN_EVAL_MAX_PAIRS,
    eval_checkpoints: list[int] | tuple[int, ...] | str | None = None,
    bootstrap_resamples: int = 400,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scores: list[float] = []
    case_time_budget_seconds = max(0.0, float(TRAINING_CASE_TIME_BUDGET_SECONDS))
    timeout_enforcement_mode = _timeout_enforcement_mode(case_time_budget_seconds)
    early_stop: dict[str, Any] = {
        "triggered": False,
        "pair_count": 0,
        "running_mean_delta": 0.0,
        "threshold": float(early_stop_mean_delta_threshold),
    }
    if timeout_enforcement_mode == "disabled_non_main_thread":
        logger.warning("Timeout guard inactive for %s: running on non-main thread", phase)
        if progress_tracker:
            progress_tracker.emit(
                event_type="timeout_guard_inactive",
                phase=phase,
                step="timeout_mode_check",
                message=f"{phase}: timeout enforcement inactive on non-main thread",
                epoch_current=epoch_current,
                payload={
                    "timeout_enforcement_mode": timeout_enforcement_mode,
                    "case_time_budget_seconds": case_time_budget_seconds,
                },
            )

    requested_cases = len(selected_cases)
    adaptive_mode = bool(ENABLE_ADAPTIVE_EVAL_V1 and str(eval_mode) == "adaptive_sequential")
    if adaptive_mode:
        max_pairs = max(1, int(eval_max_pairs))
        scheduled_cases = min(requested_cases, max_pairs)
    else:
        scheduled_cases = requested_cases
    eval_cases = selected_cases[:scheduled_cases]
    adaptive_checkpoints = _parse_eval_checkpoints(
        eval_checkpoints,
        default=DEFAULT_TRAIN_EVAL_CHECKPOINTS,
    )
    min_pairs = max(1, int(eval_min_pairs))
    adaptive_eval_trace: dict[str, Any] = {
        "mode": "adaptive_sequential" if adaptive_mode else "fixed",
        "min_pairs": int(min_pairs),
        "max_pairs": int(scheduled_cases),
        "checkpoints": adaptive_checkpoints,
        "decision": "none",
        "decision_reason": None,
        "events": [],
    }
    with prompt_override_context(prompt_suite):
        pipeline = Pipeline(
            enable_image_embedding=False,
            allow_visual_outputs=False,
        )

        for index, case in enumerate(eval_cases, start=1):
            case_prompt = str(case.get("prompt", ""))
            validation = case.get("validation")
            case_payload = {
                "id": case.get("id"),
                "category": case.get("category"),
                "difficulty": case.get("difficulty"),
                "prompt": case_prompt,
                "validation": validation,
                "answer": case.get("answer"),
            }
            case_meta = {
                "index": index,
                "total": scheduled_cases,
                "id": case_payload.get("id"),
                "category": case_payload.get("category"),
                "difficulty": case_payload.get("difficulty"),
                "prompt_preview": case_prompt[:140],
            }

            if progress_tracker:
                progress_tracker.emit(
                    event_type="case_started",
                    phase=phase,
                    step="case_run",
                    message=f"Running case {index}/{scheduled_cases}",
                    epoch_current=epoch_current,
                    current_case=case_meta,
                    payload={
                        "validation_preview": str(validation or "")[:180],
                        "timeout_enforcement_mode": timeout_enforcement_mode,
                    },
                )

            case_started_perf = time.perf_counter()
            run_elapsed_ms = 0.0
            grading_elapsed_ms = 0.0
            try:
                with _time_limit(case_time_budget_seconds):
                    run_started_perf = time.perf_counter()
                    if progress_tracker:
                        with progress_tracker.active_call(
                            "pipeline.run",
                            metadata={
                                "phase": phase,
                                "case_index": index,
                                "case_id": case.get("id"),
                            },
                        ):
                            run_output = pipeline.run(
                                prompt=case_prompt,
                                thinking_level=thinking_level,
                                include_events=False,
                            )
                    else:
                        run_output = pipeline.run(
                            prompt=case_prompt,
                            thinking_level=thinking_level,
                            include_events=False,
                        )
                    run_elapsed_ms = (time.perf_counter() - run_started_perf) * 1000.0

                    if _should_short_circuit_grading(run_output):
                        grade_started_perf = time.perf_counter()
                        grade = _short_circuit_grade(run_output)
                        grade = _postprocess_grade(
                            case=case_payload,
                            run_output=run_output,
                            grade=grade,
                            timed_out=False,
                        )
                        grading_elapsed_ms = (time.perf_counter() - grade_started_perf) * 1000.0
                    else:
                        grade_started_perf = time.perf_counter()
                        if progress_tracker:
                            with progress_tracker.active_call(
                                "grading.llm",
                                metadata={
                                    "phase": phase,
                                    "case_index": index,
                                    "case_id": case.get("id"),
                                },
                            ):
                                grade = grade_result(case_payload, run_output)
                        else:
                            grade = grade_result(case_payload, run_output)
                        grade = _postprocess_grade(
                            case=case_payload,
                            run_output=run_output,
                            grade=grade,
                            timed_out=False,
                        )
                        grading_elapsed_ms = (time.perf_counter() - grade_started_perf) * 1000.0

                case_elapsed_ms = (time.perf_counter() - case_started_perf) * 1000.0

                score = float(grade.get("aggregate_score", 0.0))
                scores.append(score)
                compact_run_output = _compact_run_output_for_storage(run_output)
                tool_metrics = _extract_case_tool_metrics(run_output)
                case_stats = {
                    "case_total_seconds": round(case_elapsed_ms / 1000.0, 6),
                    "pipeline_run_seconds": round(run_elapsed_ms / 1000.0, 6),
                    "grading_seconds": round(grading_elapsed_ms / 1000.0, 6),
                    "tool_failure_signals_count": int(tool_metrics.get("tool_failure_signals_count", 0)),
                    "degraded_mode_active": bool(tool_metrics.get("degraded_mode_active")),
                    "python_exec_attempts": int(tool_metrics.get("python_exec_attempts", 0)),
                    "python_exec_failures": int(tool_metrics.get("python_exec_failures", 0)),
                    "tool_invocations": tool_metrics.get("tool_invocations", {}),
                    "tool_errors": tool_metrics.get("tool_errors", {}),
                    "timed_out": False,
                    "case_time_budget_seconds": case_time_budget_seconds,
                    "timeout_enforcement_mode": timeout_enforcement_mode,
                }
                results.append(
                    {
                        "case": case_payload,
                        "prompt": case_prompt,
                        "validation": validation,
                        "run_output": compact_run_output,
                        "grade": grade,
                        "case_stats": case_stats,
                    }
                )
            except TimeoutError as exc:
                case_elapsed_ms = (time.perf_counter() - case_started_perf) * 1000.0
                case_stats = {
                    "case_total_seconds": round(case_elapsed_ms / 1000.0, 6),
                    "pipeline_run_seconds": None,
                    "grading_seconds": None,
                    "tool_failure_signals_count": 1,
                    "degraded_mode_active": True,
                    "python_exec_attempts": 0,
                    "python_exec_failures": 0,
                    "tool_invocations": {},
                    "tool_errors": {},
                    "timed_out": True,
                    "case_time_budget_seconds": case_time_budget_seconds,
                    "timeout_enforcement_mode": timeout_enforcement_mode,
                }
                results.append(
                    {
                        "case": case_payload,
                        "prompt": case_prompt,
                        "validation": validation,
                        "run_output": {
                            "response": "",
                            "degraded_mode_active": True,
                            "degraded_notes": [f"Case timeout: {exc}"],
                        },
                        "grade": {
                            "aggregate_score": 0.0,
                            "major_issues": f"Case timed out: {exc}",
                            "strengths": "",
                            "grading_mode": "case_timeout",
                        },
                        "case_stats": case_stats,
                    }
                )
                scores.append(0.0)
                if progress_tracker:
                    progress_tracker.emit(
                        event_type="case_completed",
                        phase=phase,
                        step="case_timeout",
                        message=f"Case {index}/{scheduled_cases} timed out",
                        epoch_current=epoch_current,
                        current_case=case_meta,
                        payload={
                            "error": str(exc),
                            "degraded_mode_active": True,
                            "tool_failure_signals_count": 1,
                            "python_exec_attempts": 0,
                            "python_exec_failures": 0,
                            "timed_out": True,
                            "case_time_budget_seconds": case_time_budget_seconds,
                            "timeout_enforcement_mode": timeout_enforcement_mode,
                            "timing_ms": {
                                "case_total": round(case_elapsed_ms, 2),
                            },
                        },
                        metrics={"latest_case_score": 0.0},
                    )
                continue
            except Exception as exc:
                case_elapsed_ms = (time.perf_counter() - case_started_perf) * 1000.0
                case_stats = {
                    "case_total_seconds": round(case_elapsed_ms / 1000.0, 6),
                    "pipeline_run_seconds": None,
                    "grading_seconds": None,
                    "tool_failure_signals_count": 1,
                    "degraded_mode_active": True,
                    "python_exec_attempts": 0,
                    "python_exec_failures": 0,
                    "tool_invocations": {},
                    "tool_errors": {},
                    "timed_out": False,
                    "case_time_budget_seconds": case_time_budget_seconds,
                    "timeout_enforcement_mode": timeout_enforcement_mode,
                }
                results.append(
                    {
                        "case": case_payload,
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
                        "case_stats": case_stats,
                    }
                )
                scores.append(0.0)
                if progress_tracker:
                    progress_tracker.emit(
                        event_type="case_completed",
                        phase=phase,
                        step="case_error",
                        message=f"Case {index}/{scheduled_cases} failed",
                        epoch_current=epoch_current,
                        current_case=case_meta,
                        payload={
                            "error": str(exc),
                            "degraded_mode_active": True,
                            "tool_failure_signals_count": 1,
                            "python_exec_attempts": 0,
                            "python_exec_failures": 0,
                            "timed_out": False,
                            "case_time_budget_seconds": case_time_budget_seconds,
                            "timeout_enforcement_mode": timeout_enforcement_mode,
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
                    key: grade.get(key)
                    for key in _GRADE_WEIGHTS
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
                tool_metrics = _extract_case_tool_metrics(run_output)
                progress_tracker.emit(
                    event_type="case_completed",
                    phase=phase,
                    step="case_scored",
                    message=f"Case {index}/{scheduled_cases} scored",
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
                        "tool_failure_signals_count": int(tool_metrics.get("tool_failure_signals_count", 0))
                        if isinstance(failure_signals, list)
                        else int(tool_metrics.get("tool_failure_signals_count", 0)),
                        "python_exec_attempts": int(tool_metrics.get("python_exec_attempts", 0)),
                        "python_exec_failures": int(tool_metrics.get("python_exec_failures", 0)),
                        "tool_invocations": tool_metrics.get("tool_invocations", {}),
                        "tool_errors": tool_metrics.get("tool_errors", {}),
                        "step_coverage_ratio": step_coverage_ratio,
                        "citation_count": len(citation_links) if isinstance(citation_links, dict) else 0,
                        "image_path_count": len(image_paths) if isinstance(image_paths, list) else 0,
                        "embedded_image_count": len(image_embeddings) if isinstance(image_embeddings, list) else 0,
                        "timed_out": False,
                        "case_time_budget_seconds": case_time_budget_seconds,
                        "timeout_enforcement_mode": timeout_enforcement_mode,
                        "deterministic_answer_score": grade.get("deterministic_answer_score"),
                        "reliability_penalty": grade.get("reliability_penalty"),
                    },
                    metrics={"latest_case_score": scores[-1]},
                )

            if reference_scores:
                pair_count = min(len(reference_scores), len(scores))
                checkpoint_hit = pair_count in set(adaptive_checkpoints)
                if adaptive_mode and checkpoint_hit:
                    delta_stats = _paired_delta_stats(
                        list(reference_scores[:pair_count]),
                        list(scores[:pair_count]),
                        bootstrap_resamples=max(100, int(bootstrap_resamples)),
                        rng=random.Random((pair_count * 7919) + int(epoch_current or 0)),
                    )
                    deltas = [float(value) for value in (delta_stats.get("deltas") or [])]
                    wins = sum(1 for value in deltas if value > 0.0)
                    win_rate = (float(wins) / float(len(deltas))) if deltas else 0.0
                    catastrophic_regression_count = sum(
                        1 for value in deltas if value <= float(TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD)
                    )
                    running_mean_delta = float(delta_stats.get("mean_delta") or 0.0)
                    ci_lower = float(delta_stats.get("ci_lower") or running_mean_delta)
                    ci_upper = float(delta_stats.get("ci_upper") or running_mean_delta)
                    event_payload = {
                        "pair_count": pair_count,
                        "running_mean_delta": running_mean_delta,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "win_rate": win_rate,
                        "catastrophic_regression_count": catastrophic_regression_count,
                    }
                    adaptive_eval_trace["events"].append(dict(event_payload))
                    early_reject_reason = None
                    if pair_count >= 6:
                        if ci_upper < 0.0:
                            early_reject_reason = "ci_upper_below_zero"
                        elif catastrophic_regression_count >= 2:
                            early_reject_reason = "catastrophic_regressions_ge_2"
                        elif running_mean_delta < -0.75:
                            early_reject_reason = "running_mean_delta_below_-0.75"
                    early_accept = (
                        pair_count >= 6
                        and ci_lower > 0.20
                        and win_rate >= 0.60
                        and catastrophic_regression_count == 0
                    )
                    if early_reject_reason:
                        adaptive_eval_trace["decision"] = "early_reject"
                        adaptive_eval_trace["decision_reason"] = early_reject_reason
                        early_stop = {
                            "triggered": True,
                            "pair_count": pair_count,
                            "running_mean_delta": running_mean_delta,
                            "threshold": float(early_stop_mean_delta_threshold),
                            "reason": early_reject_reason,
                        }
                        if progress_tracker:
                            progress_tracker.emit(
                                event_type="phase_early_stopped",
                                phase=phase,
                                step="adaptive_early_reject",
                                message=f"{phase}: adaptive early reject at checkpoint {pair_count}",
                                epoch_current=epoch_current,
                                payload={**early_stop, **event_payload},
                            )
                        break
                    if early_accept:
                        adaptive_eval_trace["decision"] = "early_accept"
                        adaptive_eval_trace["decision_reason"] = "ci_lower_winrate_no_catastrophic"
                        if progress_tracker:
                            progress_tracker.emit(
                                event_type="phase_early_stopped",
                                phase=phase,
                                step="adaptive_early_accept",
                                message=f"{phase}: adaptive early accept at checkpoint {pair_count}",
                                epoch_current=epoch_current,
                                payload=event_payload,
                            )
                        break

                if not adaptive_mode:
                    triggered, stop_details = _should_early_stop_candidate_eval(
                        baseline_scores=reference_scores,
                        candidate_scores=scores,
                        min_cases=early_stop_min_cases,
                        mean_delta_threshold=early_stop_mean_delta_threshold,
                    )
                    if triggered:
                        early_stop = {"triggered": True, **stop_details}
                        if progress_tracker:
                            progress_tracker.emit(
                                event_type="phase_early_stopped",
                                phase=phase,
                                step="early_stop",
                                message=(
                                    f"{phase}: early stop triggered after {int(stop_details.get('pair_count', 0))} case(s)"
                                ),
                                epoch_current=epoch_current,
                                payload=early_stop,
                            )
                        break

    if adaptive_mode and str(adaptive_eval_trace.get("decision")) == "none":
        adaptive_eval_trace["decision"] = "completed"
        adaptive_eval_trace["decision_reason"] = (
            "max_pairs_reached" if len(scores) >= int(scheduled_cases) else "completed_without_checkpoint_stop"
        )

    return results, scores, {
        "case_count_requested": requested_cases,
        "case_count_scheduled": scheduled_cases,
        "case_count_evaluated": len(scores),
        "timeout_enforcement_mode": timeout_enforcement_mode,
        "early_stop": early_stop,
        "evaluation_mode": "adaptive_sequential" if adaptive_mode else "fixed",
        "adaptive_eval_trace": adaptive_eval_trace,
    }


def train_ab_loop(
    base_prompts_path: str,
    output_path: str,
    test_cases_path: str,
    epochs: int = 10,
    num_test_cases_per_trial: int = 10,
    random_seed: int = 42,
    thinking_level: Literal["low", "med-synth", "med-plan", "high"] = "med-synth",
    fail_threshold: float | None = None,
    holdout_sample_size: int = 6,
    rca_case_budget: int = 3,
    mutation_block_budget: int = 1,
    mutation_accept_threshold: int = MUTATION_ACCEPTANCE_TOTAL_MIN,
    mutation_tournament_size: int = DEFAULT_MUTATION_TOURNAMENT_SIZE,
    mutation_retry_enabled: bool = True,
    mutation_max_retries: int = DEFAULT_MUTATION_MAX_RETRIES,
    generalizer_cadence: int = 3,
    generalizer_suspicious_delta_threshold: int = GENERALIZER_SUSPICIOUS_DELTA_THRESHOLD,
    bootstrap_resamples: int = 1000,
    rca_threshold: float = DEFAULT_RCA_THRESHOLD,
    rca_fallback_fraction: float = DEFAULT_RCA_FALLBACK_FRACTION,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    runtime_gate_mean_ratio_max: float = RUNTIME_GATE_MEAN_MAX_RATIO,
    runtime_gate_p90_ratio_max: float = RUNTIME_GATE_P90_MAX_RATIO,
    runtime_gate_abs_mean_increase_max_seconds: float = RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS,
    stability_gate_tool_failure_delta_max: float = STABILITY_GATE_MAX_TOOL_FAILURE_DELTA,
    stability_gate_degraded_rate_delta_max: float = DEGRADATION_GATE_MAX_DELTA,
    holdout_winrate_min: float = HOLDOUT_GATE_MIN_WIN_RATE,
    quality_gate_min_mean_delta: float = QUALITY_GATE_MIN_MEAN_DELTA,
    quality_gate_min_win_rate: float = QUALITY_GATE_MIN_WIN_RATE,
    quality_gate_min_p10_delta: float = QUALITY_GATE_MIN_P10_DELTA,
    quality_gate_min_worst_case_delta: float = QUALITY_GATE_MIN_WORST_CASE_DELTA,
    tail_catastrophic_threshold: float = TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD,
    tail_max_catastrophic_regressions: int = TAIL_RISK_MAX_CATASTROPHIC_REGRESSIONS,
    eval_mode: str = DEFAULT_EVAL_MODE,
    train_eval_min_pairs: int = DEFAULT_TRAIN_EVAL_MIN_PAIRS,
    train_eval_max_pairs: int = DEFAULT_TRAIN_EVAL_MAX_PAIRS,
    train_eval_checkpoints: list[int] | tuple[int, ...] | str | None = None,
    holdout_eval_min_pairs: int = DEFAULT_HOLDOUT_EVAL_MIN_PAIRS,
    holdout_eval_max_pairs: int = DEFAULT_HOLDOUT_EVAL_MAX_PAIRS,
    holdout_eval_checkpoints: list[int] | tuple[int, ...] | str | None = None,
    mutation_stage_a_top_k: int = DEFAULT_MUTATION_STAGE_A_TOP_K,
    mutation_precheck_cases: int = DEFAULT_MUTATION_PRECHECK_CASES,
    resource_target_reduction: float = DEFAULT_RESOURCE_TARGET_REDUCTION,
    holdout_split_ratio: float = DEFAULT_HOLDOUT_SPLIT_RATIO,
    progress_status_path: str = "data/training_status.json",
    progress_events_path: str = "data/training_events.jsonl",
    progress_events_archive_dir: str = "data/training_events",
    track_progress: bool = True,
) -> dict[str, Any]:
    rng = random.Random(random_seed)

    current_prompts = load_prompts_file(base_prompts_path)
    test_cases, dataset_diagnostics = load_test_cases_with_diagnostics(test_cases_path)
    effective_rca_threshold = float(fail_threshold) if fail_threshold is not None else float(rca_threshold)
    train_pool, holdout_pool = stratified_split_cases(
        test_cases,
        holdout_ratio=holdout_split_ratio,
        rng=rng,
    )
    if not train_pool:
        train_pool = list(test_cases)
        holdout_pool = []

    train_sampler_state = _build_sampling_state(train_pool, rng)
    holdout_sampler_state = _build_sampling_state(holdout_pool, rng) if holdout_pool else {"buckets": {}}
    block_details_map = get_prompted_block_details()

    # Mutation coverage tracking (Phase 4C)
    mutation_coverage: dict[str, dict[str, int]] = {}
    case_history: dict[str, dict[str, Any]] = {}
    block_impact_history: dict[str, dict[str, Any]] = {}

    store = PromptSuiteStore(output_path)
    prior_generations = store.data.get("generations", []) if isinstance(store.data, dict) else []
    if isinstance(prior_generations, list):
        for record in reversed(prior_generations):
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            snapshot = metadata.get("case_history_snapshot")
            if isinstance(snapshot, dict) and snapshot:
                case_history = {
                    str(case_id): dict(history)
                    for case_id, history in snapshot.items()
                    if isinstance(history, dict)
                }
            block_snapshot = metadata.get("block_impact_history_snapshot")
            if isinstance(block_snapshot, dict) and block_snapshot:
                block_impact_history = {
                    str(block_id): dict(history)
                    for block_id, history in block_snapshot.items()
                    if isinstance(history, dict)
                }
            if case_history or block_impact_history:
                break
    run_id = str(uuid.uuid4())
    events_archive_path = str((Path(progress_events_archive_dir) / f"{run_id}.jsonl").resolve())
    generation = 0
    split_metadata = {
        "train_split": {
            "size": len(train_pool),
            "case_ids": [_case_identifier(case) for case in train_pool],
        },
        "holdout_split": {
            "size": len(holdout_pool),
            "case_ids": [_case_identifier(case) for case in holdout_pool],
            "ratio": max(0.0, min(0.5, float(holdout_split_ratio))),
        },
    }
    store.save_generation(
        current_prompts,
        generation,
        {
            "note": "initial",
            "run_id": run_id,
            "events_archive_path": events_archive_path,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "dataset_diagnostics": dataset_diagnostics,
            **split_metadata,
        },
    )
    tracker = TrainingProgressTracker(
        status_path=progress_status_path,
        events_path=progress_events_path,
        events_archive_path=events_archive_path,
        run_id=run_id,
        epochs_total=epochs,
        enabled=track_progress,
    )
    resource_telemetry = RunResourceTelemetry() if ENABLE_RESOURCE_TELEMETRY_V3 else None
    resource_baseline_snapshot = resource_telemetry.snapshot() if resource_telemetry is not None else {}

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
            "holdout_sample_size": holdout_sample_size,
            "thinking_level": thinking_level,
            "fail_threshold": fail_threshold,
            "rca_threshold": float(effective_rca_threshold),
            "rca_fallback_fraction": float(rca_fallback_fraction),
            "selection_mode": str(selection_mode),
            "random_seed": random_seed,
            "total_available_cases": len(test_cases),
            "sample_strategy": str(selection_mode),
            "rca_case_budget": int(max(0, rca_case_budget)),
            "mutation_block_budget": int(max(0, mutation_block_budget)),
            "mutation_accept_threshold": int(max(1, mutation_accept_threshold)),
            "generalizer_cadence": int(max(0, generalizer_cadence)),
            "generalizer_suspicious_delta_threshold": int(max(0, generalizer_suspicious_delta_threshold)),
            "mutation_retry_enabled": bool(mutation_retry_enabled),
            "mutation_max_retries": int(max(0, mutation_max_retries)),
            "bootstrap_resamples": int(max(1, bootstrap_resamples)),
            "train_split_size": len(train_pool),
            "holdout_split_size": len(holdout_pool),
            "dataset_diagnostics": dataset_diagnostics,
            "case_history_seeded_count": len(case_history),
            "runtime_gate_mean_ratio_max": float(runtime_gate_mean_ratio_max),
            "runtime_gate_p90_ratio_max": float(runtime_gate_p90_ratio_max),
            "runtime_gate_abs_mean_increase_max_seconds": float(runtime_gate_abs_mean_increase_max_seconds),
            "stability_gate_tool_failure_delta_max": float(stability_gate_tool_failure_delta_max),
            "stability_gate_degraded_rate_delta_max": float(stability_gate_degraded_rate_delta_max),
            "holdout_winrate_min": float(holdout_winrate_min),
            "quality_gate_min_mean_delta": float(quality_gate_min_mean_delta),
            "quality_gate_min_win_rate": float(quality_gate_min_win_rate),
            "quality_gate_min_p10_delta": float(quality_gate_min_p10_delta),
            "quality_gate_min_worst_case_delta": float(quality_gate_min_worst_case_delta),
            "tail_catastrophic_threshold": float(tail_catastrophic_threshold),
            "tail_max_catastrophic_regressions": int(max(0, tail_max_catastrophic_regressions)),
            "eval_mode": str(eval_mode),
            "train_eval_min_pairs": int(max(1, train_eval_min_pairs)),
            "train_eval_max_pairs": int(max(1, train_eval_max_pairs)),
            "train_eval_checkpoints": _parse_eval_checkpoints(
                train_eval_checkpoints,
                default=DEFAULT_TRAIN_EVAL_CHECKPOINTS,
            ),
            "holdout_eval_min_pairs": int(max(1, holdout_eval_min_pairs)),
            "holdout_eval_max_pairs": int(max(1, holdout_eval_max_pairs)),
            "holdout_eval_checkpoints": _parse_eval_checkpoints(
                holdout_eval_checkpoints,
                default=DEFAULT_HOLDOUT_EVAL_CHECKPOINTS,
            ),
            "mutation_stage_a_top_k": int(max(1, mutation_stage_a_top_k)),
            "mutation_precheck_cases": int(max(0, mutation_precheck_cases)),
            "resource_target_reduction": float(max(0.0, resource_target_reduction)),
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "events_archive_path": events_archive_path,
            "feature_flags": {
                "ENABLE_CANDIDATE_GATES": ENABLE_CANDIDATE_GATES,
                "ENABLE_TOOL_PROMPT_TEMPLATES": ENABLE_TOOL_PROMPT_TEMPLATES,
                "ENABLE_MUTATION_RETRY_V2": ENABLE_MUTATION_RETRY_V2,
                "ENABLE_TAIL_RISK_GATE": ENABLE_TAIL_RISK_GATE,
                "ENABLE_NO_CHANGE_STRICT_SEMANTICS": ENABLE_NO_CHANGE_STRICT_SEMANTICS,
                "ENABLE_BLOCK_IMPACT_RANKER": ENABLE_BLOCK_IMPACT_RANKER,
                "ENABLE_RESOURCE_TELEMETRY_V3": ENABLE_RESOURCE_TELEMETRY_V3,
                "ENABLE_ADAPTIVE_EVAL_V1": ENABLE_ADAPTIVE_EVAL_V1,
                "ENABLE_BALANCED_GUARDRAILS_V1": ENABLE_BALANCED_GUARDRAILS_V1,
                "ENABLE_MUTATION_BUDGET_OPT_V1": ENABLE_MUTATION_BUDGET_OPT_V1,
                "ENABLE_REPLAY_REBALANCE_V1": ENABLE_REPLAY_REBALANCE_V1,
                "TRAINING_CASE_TIME_BUDGET_SECONDS": TRAINING_CASE_TIME_BUDGET_SECONDS,
                "STALL_WARNING_THRESHOLD_SECONDS": STALL_WARNING_THRESHOLD_SECONDS,
            },
        },
    )

    global _ACTIVE_RESOURCE_TELEMETRY
    previous_resource_telemetry = _ACTIVE_RESOURCE_TELEMETRY
    _ACTIVE_RESOURCE_TELEMETRY = resource_telemetry if ENABLE_RESOURCE_TELEMETRY_V3 else None

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

            sample_budget = min(num_test_cases_per_trial, len(train_pool))
            selection_diagnostics: list[dict[str, Any]] = []
            if str(selection_mode) == "hybrid_coverage_replay":
                selected, selection_diagnostics = select_cases_hybrid(
                    sample_state=train_sampler_state,
                    sample_size=sample_budget,
                    rng=rng,
                    case_history=case_history,
                    epoch_current=epoch_current,
                )
            else:
                selected = sample_cases_stratified(
                    sample_state=train_sampler_state,
                    sample_size=sample_budget,
                    rng=rng,
                )
                selection_diagnostics = [
                    {
                        "case_id": _case_identifier(case),
                        "slot": "coverage",
                        "reason": "stratified_rotation",
                        "category": case.get("category"),
                        "difficulty": case.get("difficulty"),
                    }
                    for case in selected
                ]
            rng.shuffle(selected)
            tracker.emit(
                event_type="sample_selected",
                phase="sampling",
                step="select_cases",
                message=f"Selected {len(selected)} case(s) for epoch {epoch_current}",
                epoch_current=epoch_current,
                payload={
                    "sampled_case_ids": [case.get("id") for case in selected],
                    "sampled_categories": sorted({case.get("category") for case in selected if case.get("category")}),
                    "sample_strategy": str(selection_mode),
                    "selection_diagnostics": selection_diagnostics,
                    "pool": "train",
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
            if resource_telemetry is not None:
                with resource_telemetry.phase("phase_a_eval"):
                    results_a, scores_a, phase_a_meta = evaluate_suite(
                        current_prompts,
                        selected,
                        thinking_level,
                        phase="phase_a_eval",
                        epoch_current=epoch_current,
                        progress_tracker=tracker,
                        eval_mode=eval_mode,
                        eval_min_pairs=train_eval_min_pairs,
                        eval_max_pairs=train_eval_max_pairs,
                        eval_checkpoints=train_eval_checkpoints,
                        bootstrap_resamples=bootstrap_resamples,
                    )
            else:
                results_a, scores_a, phase_a_meta = evaluate_suite(
                    current_prompts,
                    selected,
                    thinking_level,
                    phase="phase_a_eval",
                    epoch_current=epoch_current,
                    progress_tracker=tracker,
                    eval_mode=eval_mode,
                    eval_min_pairs=train_eval_min_pairs,
                    eval_max_pairs=train_eval_max_pairs,
                    eval_checkpoints=train_eval_checkpoints,
                    bootstrap_resamples=bootstrap_resamples,
                )
            avg_score_a = sum(scores_a) / max(len(scores_a), 1)
            stats_a = _score_stats(scores_a)
            phase_a_eval_summary = _summarize_eval_results(results_a)
            tracker.emit(
                event_type="phase_completed",
                phase="phase_a_eval",
                step="evaluate_baseline_done",
                message=f"Epoch {epoch_current}: baseline evaluation completed",
                epoch_current=epoch_current,
                payload={
                    "score_stats": stats_a,
                    "eval_summary": phase_a_eval_summary,
                    "evaluation_meta": phase_a_meta,
                },
                metrics={
                    "avg_score_a": avg_score_a,
                    "phase_a_min_score": stats_a["min"],
                    "phase_a_max_score": stats_a["max"],
                    "phase_a_median_score": stats_a["median"],
                    "phase_a_mean_case_time_s": phase_a_eval_summary.get("mean_case_time_s", 0.0),
                    "phase_a_p90_case_time_s": phase_a_eval_summary.get("p90_case_time_s", 0.0),
                    "phase_a_avg_tool_failure_signals": phase_a_eval_summary.get("avg_tool_failure_signals", 0.0),
                    "phase_a_degraded_case_rate": phase_a_eval_summary.get("degraded_case_rate", 0.0),
                    "phase_a_avg_python_exec_attempts": phase_a_eval_summary.get("avg_python_exec_attempts", 0.0),
                    "phase_a_avg_python_exec_failures": phase_a_eval_summary.get("avg_python_exec_failures", 0.0),
                },
            )

            failed, rca_selection_diagnostics = _select_rca_cases(
                results_a,
                rca_case_budget=rca_case_budget,
                rca_threshold=effective_rca_threshold,
                rca_fallback_fraction=rca_fallback_fraction,
                case_history=case_history,
            )
            all_analyses: list[BlockAnalysisSchema] = []
            valid_block_ids = sorted(list(current_prompts.keys()))
            high_level_rca: dict[str, Any] = {"summary": "No RCA themes generated", "themes": []}
            rca_case_ids = [
                _case_identifier(item.get("case", {}))
                for item in failed
                if isinstance(item.get("case"), dict)
            ]
            tracker.emit(
                event_type="rca_batch_started",
                phase="rca",
                step="identify_failures",
                message=f"Epoch {epoch_current}: running RCA on {len(failed)} failed case(s)",
                epoch_current=epoch_current,
                payload={
                    "failed_cases": len(failed),
                    "rca_case_budget": int(max(0, rca_case_budget)),
                    "rca_threshold": float(effective_rca_threshold),
                    "rca_fallback_fraction": float(rca_fallback_fraction),
                    "rca_case_ids": rca_case_ids,
                    "rca_selection_diagnostics": rca_selection_diagnostics,
                    "failed_case_prompts": [
                        str(item.get("prompt", ""))[:180]
                        for item in failed[:20]
                    ],
                },
            )

            if failed:
                tracker.emit(
                    event_type="rca_high_level_started",
                    phase="rca",
                    step="cohort_theme_rca",
                    message=f"Epoch {epoch_current}: running high-level RCA themes",
                    epoch_current=epoch_current,
                    payload={
                        "cohort_size": len(failed),
                        "case_ids": rca_case_ids,
                        "selection_rule": rca_selection_diagnostics.get("rule"),
                    },
                )
                high_level_rca = run_high_level_rca(
                    failed_results=failed,
                    valid_blocks=valid_block_ids,
                )
                tracker.emit(
                    event_type="rca_high_level_completed",
                    phase="rca",
                    step="cohort_theme_rca_done",
                    message=f"Epoch {epoch_current}: high-level RCA completed",
                    epoch_current=epoch_current,
                    payload={
                        "theme_count": len(high_level_rca.get("themes") or []),
                        "summary": high_level_rca.get("summary"),
                        "themes": high_level_rca.get("themes"),
                        "model_used": high_level_rca.get("model_used"),
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

                case_payload = failed_case.get("case", {})
                case_context = {
                    "id": case_payload.get("id") if isinstance(case_payload, dict) else None,
                    "category": case_payload.get("category") if isinstance(case_payload, dict) else None,
                    "difficulty": case_payload.get("difficulty") if isinstance(case_payload, dict) else None,
                    "aggregate_score": float(failed_case.get("grade", {}).get("aggregate_score", 0.0)),
                    "degraded_mode_active": bool(failed_case.get("case_stats", {}).get("degraded_mode_active")),
                    "timed_out": bool(failed_case.get("case_stats", {}).get("timed_out")),
                }
                implicated_block_ids = _implicated_blocks_for_case(
                    failed_case.get("run_output", {}),
                    valid_block_ids,
                )
                criteria_context_text = _criteria_context_for_blocks(implicated_block_ids, block_details_map)
                analyses: list[BlockAnalysisSchema]
                if tracker is not None:
                    with tracker.active_call(
                        "rca.llm",
                        metadata={
                            "epoch": epoch_current,
                            "failed_index": failed_index,
                            "failed_total": len(failed),
                            "implicated_blocks": implicated_block_ids[:12],
                        },
                    ):
                        analyses = root_cause_analysis(
                            prompt=failed_case["prompt"],
                            run_output=failed_case["run_output"],
                            major_issues=str(failed_case["grade"].get("major_issues", "")),
                            valid_blocks=implicated_block_ids,
                            block_details_map=block_details_map,
                            criteria_context_text=criteria_context_text,
                            case_context=case_context,
                            high_level_themes=list(high_level_rca.get("themes") or []),
                        )
                else:
                    analyses = root_cause_analysis(
                        prompt=failed_case["prompt"],
                        run_output=failed_case["run_output"],
                        major_issues=str(failed_case["grade"].get("major_issues", "")),
                        valid_blocks=implicated_block_ids,
                        block_details_map=block_details_map,
                        criteria_context_text=criteria_context_text,
                        case_context=case_context,
                        high_level_themes=list(high_level_rca.get("themes") or []),
                    )
                all_analyses.extend(analyses)
                case_theme_links = _dedupe_non_empty_strings(
                    [theme_id for item in analyses for theme_id in (item.linked_theme_ids or [])],
                    max_items=12,
                )
                tracker.emit(
                    event_type="rca_completed",
                    phase="rca",
                    step="case_rca_done",
                    message=f"Epoch {epoch_current}: RCA {failed_index}/{len(failed)} complete",
                    epoch_current=epoch_current,
                    payload={
                        "analyses_added": len(analyses),
                        "implicated_blocks": implicated_block_ids,
                        "linked_theme_ids": case_theme_links,
                        "recommended_fix_patterns": _dedupe_non_empty_strings(
                            [pattern for item in analyses for pattern in (item.recommended_fix_patterns or [])],
                            max_items=10,
                        ),
                    },
                )

            candidate_prompts = dict(current_prompts)
            tracker.emit(
                event_type="mutation_started",
                phase="prompt_improvement",
                step="generate_mutations",
                message=f"Epoch {epoch_current}: generating prompt mutations",
                epoch_current=epoch_current,
                payload={
                    "rca_items": len(all_analyses),
                    "mutation_block_budget": int(max(0, mutation_block_budget)),
                    "mutation_block_budget_deprecated": True,
                    "mutation_accept_threshold": int(max(1, mutation_accept_threshold)),
                    "mutation_retry_enabled": bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                    "mutation_max_retries": int(max(0, mutation_max_retries)),
                    "mutation_style_catalog": list(MUTATION_STYLE_DEFAULT_ORDER),
                    "mutation_diversity_target": int(DEFAULT_MUTATION_DIVERSITY_TARGET),
                    "mutation_stage_a_top_k": int(max(1, mutation_stage_a_top_k)),
                    "mutation_precheck_cases": int(max(0, mutation_precheck_cases)),
                    "mutation_budget_mode": "staged_v1" if ENABLE_MUTATION_BUDGET_OPT_V1 else "legacy_full",
                },
            )
            grouped_stage: dict[str, list[BlockAnalysisSchema]] = defaultdict(list)
            for analysis in all_analyses:
                if analysis.need_fix:
                    grouped_stage[analysis.block_id].append(analysis)
            ranked_stage_ids = [
                block_id
                for _, block_id in _rank_blocks_by_rca_impact(
                    grouped_stage,
                    block_impact_history=block_impact_history,
                    enable_block_impact_ranker=ENABLE_BLOCK_IMPACT_RANKER,
                )
            ]
            stage_a_count = int(max(1, mutation_stage_a_top_k))
            stage_a_ids = ranked_stage_ids[:stage_a_count] if ranked_stage_ids else []
            stage_b_ids = ranked_stage_ids[stage_a_count:] if ranked_stage_ids else []

            def _run_prompt_improvements_once(
                *,
                suite: dict[str, str],
                target_ids: list[str] | None,
                stage_label: str,
            ) -> tuple[dict[str, str], list[dict[str, Any]]]:
                if resource_telemetry is not None:
                    with resource_telemetry.phase(f"mutation_{stage_label}"):
                        if tracker is not None:
                            with tracker.active_call(
                                "prompt_improvements",
                                metadata={
                                    "epoch": epoch_current,
                                    "rca_items": len(all_analyses),
                                    "stage": stage_label,
                                    "target_blocks": target_ids or [],
                                },
                            ):
                                return prompt_improvements(
                                    current_prompts=suite,
                                    block_analyses=all_analyses,
                                    block_details_map=block_details_map,
                                    mutation_block_budget=mutation_block_budget,
                                    mutation_accept_threshold=mutation_accept_threshold,
                                    mutation_tournament_size=mutation_tournament_size,
                                    mutation_retry_enabled=bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                                    mutation_max_retries=mutation_max_retries,
                                    block_impact_history=block_impact_history,
                                    progress_tracker=tracker,
                                    epoch_current=epoch_current,
                                    target_block_ids_override=target_ids,
                                )
                        return prompt_improvements(
                            current_prompts=suite,
                            block_analyses=all_analyses,
                            block_details_map=block_details_map,
                            mutation_block_budget=mutation_block_budget,
                            mutation_accept_threshold=mutation_accept_threshold,
                            mutation_tournament_size=mutation_tournament_size,
                            mutation_retry_enabled=bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                            mutation_max_retries=mutation_max_retries,
                            block_impact_history=block_impact_history,
                            progress_tracker=None,
                            epoch_current=epoch_current,
                            target_block_ids_override=target_ids,
                        )
                if tracker is not None:
                    with tracker.active_call(
                        "prompt_improvements",
                        metadata={
                            "epoch": epoch_current,
                            "rca_items": len(all_analyses),
                            "stage": stage_label,
                            "target_blocks": target_ids or [],
                        },
                    ):
                        return prompt_improvements(
                            current_prompts=suite,
                            block_analyses=all_analyses,
                            block_details_map=block_details_map,
                            mutation_block_budget=mutation_block_budget,
                            mutation_accept_threshold=mutation_accept_threshold,
                            mutation_tournament_size=mutation_tournament_size,
                            mutation_retry_enabled=bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                            mutation_max_retries=mutation_max_retries,
                            block_impact_history=block_impact_history,
                            progress_tracker=tracker,
                            epoch_current=epoch_current,
                            target_block_ids_override=target_ids,
                        )
                return prompt_improvements(
                    current_prompts=suite,
                    block_analyses=all_analyses,
                    block_details_map=block_details_map,
                    mutation_block_budget=mutation_block_budget,
                    mutation_accept_threshold=mutation_accept_threshold,
                    mutation_tournament_size=mutation_tournament_size,
                    mutation_retry_enabled=bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                    mutation_max_retries=mutation_max_retries,
                    block_impact_history=block_impact_history,
                    progress_tracker=None,
                    epoch_current=epoch_current,
                    target_block_ids_override=target_ids,
                )

            stage_a_improvements, stage_a_prompt_scoring = _run_prompt_improvements_once(
                suite=current_prompts,
                target_ids=stage_a_ids if ENABLE_MUTATION_BUDGET_OPT_V1 else None,
                stage_label="stage_a",
            )
            candidate_prompts.update(stage_a_improvements)
            prompt_scoring = list(stage_a_prompt_scoring)

            stage_a_accepted = any(bool(item.get("accepted")) for item in stage_a_prompt_scoring if isinstance(item, dict))
            ran_stage_b = False
            if ENABLE_MUTATION_BUDGET_OPT_V1 and stage_b_ids and not stage_a_accepted:
                ran_stage_b = True
                tracker.emit(
                    event_type="mutation_stage_expanded",
                    phase="prompt_improvement",
                    step="expand_stage_b",
                    message=f"Epoch {epoch_current}: expanding mutation to Stage B blocks",
                    epoch_current=epoch_current,
                    payload={
                        "stage_a_top_k": stage_a_count,
                        "stage_a_accepted": bool(stage_a_accepted),
                        "stage_b_block_count": len(stage_b_ids),
                    },
                )
                stage_b_improvements, stage_b_prompt_scoring = _run_prompt_improvements_once(
                    suite=candidate_prompts,
                    target_ids=stage_b_ids,
                    stage_label="stage_b",
                )
                candidate_prompts.update(stage_b_improvements)
                prompt_scoring.extend(stage_b_prompt_scoring)

            changed_keys = [
                key for key in candidate_prompts.keys() if candidate_prompts.get(key) != current_prompts.get(key)
            ]
            changed_blocks = [item for item in prompt_scoring if item.get("changed")]
            scored_blocks = [
                item
                for item in changed_blocks
                if isinstance(item.get("baseline_total"), (int, float))
                and isinstance(item.get("candidate_total"), (int, float))
            ]
            accepted_blocks = [item for item in scored_blocks if item.get("accepted")]
            rejected_blocks = [item for item in changed_blocks if not item.get("accepted")]
            contract_rejected_blocks = [
                item
                for item in changed_blocks
                if str(item.get("decision_reason", "")).endswith("prompt_contract_violation")
                or str(item.get("decision_reason", "")) == "prompt_contract_violation"
            ]
            score_rejected_blocks = [
                item
                for item in changed_blocks
                if item.get("decision_reason")
                in {"rejected_below_absolute_threshold"}
            ]
            rejection_breakdown: dict[str, int] = {}
            rejection_issue_matrix: dict[str, dict[str, int]] = {}
            for item in rejected_blocks:
                reason = str(item.get("decision_reason") or "unknown_rejection")
                rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
                block_id = str(item.get("block_id") or "unknown_block")
                issue_bucket = rejection_issue_matrix.setdefault(block_id, {})
                issues = item.get("contract_hard_issues") or item.get("contract_issues") or []
                if not isinstance(issues, list):
                    issues = []
                if not issues:
                    issue_bucket["no_contract_issue"] = issue_bucket.get("no_contract_issue", 0) + 1
                else:
                    for issue in issues:
                        issue_key = str(issue)
                        issue_bucket[issue_key] = issue_bucket.get(issue_key, 0) + 1
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
            mutation_style_summary = _summarize_mutation_style_diagnostics(prompt_scoring)

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
                    "accepted_count": len(accepted_blocks),
                    "rejected_count": len(rejected_blocks),
                    "scored_count": len(scored_blocks),
                    "changed_count": len(changed_blocks),
                    "contract_rejected_count": len(contract_rejected_blocks),
                    "score_rejected_count": len(score_rejected_blocks),
                    "mutation_rejection_breakdown": rejection_breakdown,
                    "mutation_rejection_issue_matrix": rejection_issue_matrix,
                    "mutation_style_summary": mutation_style_summary,
                },
                metrics={
                    "prompt_scored_blocks": len(scored_blocks),
                    "prompt_accepted_blocks": len(accepted_blocks),
                    "prompt_rejected_blocks": len(rejected_blocks),
                    "prompt_changed_blocks": len(changed_blocks),
                    "prompt_contract_rejected_blocks": len(contract_rejected_blocks),
                    "prompt_score_rejected_blocks": len(score_rejected_blocks),
                    "mutation_style_blocks": mutation_style_summary.get("blocks_with_styles", 0),
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
                    "prompt_changed_blocks": len(changed_blocks),
                    "prompt_contract_rejected_blocks": len(contract_rejected_blocks),
                    "prompt_score_rejected_blocks": len(score_rejected_blocks),
                    "mutation_rejection_breakdown": rejection_breakdown,
                    "mutation_rejection_issue_matrix": rejection_issue_matrix,
                    "mutation_style_summary": mutation_style_summary,
                },
            )
            tracker.emit(
                event_type="mutation_rejection_breakdown",
                phase="prompt_improvement",
                step="rejection_breakdown",
                message=f"Epoch {epoch_current}: mutation rejection breakdown",
                epoch_current=epoch_current,
                payload={
                    "mutation_rejection_breakdown": rejection_breakdown,
                    "mutation_rejection_issue_matrix": rejection_issue_matrix,
                },
            )

            # Update mutation coverage (Phase 4C)
            for item in prompt_scoring:
                bid = str(item.get("block_id") or "")
                if not bid:
                    continue
                if bid not in mutation_coverage:
                    mutation_coverage[bid] = {"targeted": 0, "mutations_generated": 0, "validation_passed": 0, "gates_passed": 0}
                mutation_coverage[bid]["targeted"] += 1
                if item.get("changed"):
                    mutation_coverage[bid]["mutations_generated"] += 1
                if item.get("accepted"):
                    mutation_coverage[bid]["validation_passed"] += 1

            precheck_trace: dict[str, Any] | None = None
            if changed_keys and int(max(0, mutation_precheck_cases)) > 0:
                precheck_budget = int(max(1, mutation_precheck_cases))
                paired_baseline = list(zip(results_a, scores_a, selected))
                paired_baseline.sort(
                    key=lambda item: float(
                        ((item[0] or {}).get("grade", {}) or {}).get("aggregate_score", 0.0) or 0.0
                    )
                )
                precheck_bundle = paired_baseline[:precheck_budget]
                precheck_cases = [bundle[2] for bundle in precheck_bundle if isinstance(bundle[2], dict)]
                precheck_baseline_scores = [float(bundle[1]) for bundle in precheck_bundle]

                if precheck_cases and len(precheck_baseline_scores) == len(precheck_cases):
                    tracker.emit(
                        event_type="precheck_started",
                        phase="phase_b_precheck_eval",
                        step="precheck_candidate",
                        message=f"Epoch {epoch_current}: running candidate precheck on {len(precheck_cases)} case(s)",
                        epoch_current=epoch_current,
                        payload={
                            "case_ids": [_case_identifier(case) for case in precheck_cases],
                            "baseline_scores": precheck_baseline_scores,
                        },
                    )
                    if resource_telemetry is not None:
                        with resource_telemetry.phase("phase_b_precheck_eval"):
                            precheck_results, precheck_scores, precheck_meta = evaluate_suite(
                                candidate_prompts,
                                precheck_cases,
                                thinking_level,
                                phase="phase_b_precheck_eval",
                                epoch_current=epoch_current,
                                progress_tracker=tracker,
                                reference_scores=precheck_baseline_scores,
                                early_stop_min_cases=EARLY_STOP_MIN_CASES,
                                early_stop_mean_delta_threshold=EARLY_STOP_MEAN_DELTA_THRESHOLD,
                                eval_mode=eval_mode,
                                eval_min_pairs=min(4, len(precheck_cases)),
                                eval_max_pairs=len(precheck_cases),
                                eval_checkpoints=[min(4, len(precheck_cases)), len(precheck_cases)],
                                bootstrap_resamples=bootstrap_resamples,
                            )
                    else:
                        precheck_results, precheck_scores, precheck_meta = evaluate_suite(
                            candidate_prompts,
                            precheck_cases,
                            thinking_level,
                            phase="phase_b_precheck_eval",
                            epoch_current=epoch_current,
                            progress_tracker=tracker,
                            reference_scores=precheck_baseline_scores,
                            early_stop_min_cases=EARLY_STOP_MIN_CASES,
                            early_stop_mean_delta_threshold=EARLY_STOP_MEAN_DELTA_THRESHOLD,
                            eval_mode=eval_mode,
                            eval_min_pairs=min(4, len(precheck_cases)),
                            eval_max_pairs=len(precheck_cases),
                            eval_checkpoints=[min(4, len(precheck_cases)), len(precheck_cases)],
                            bootstrap_resamples=bootstrap_resamples,
                        )

                    precheck_delta_stats = _paired_delta_stats(
                        precheck_baseline_scores[: len(precheck_scores)],
                        precheck_scores,
                        bootstrap_resamples=bootstrap_resamples,
                        rng=rng,
                    )
                    precheck_deltas = [float(value) for value in (precheck_delta_stats.get("deltas") or [])]
                    precheck_catastrophic_count = sum(
                        1 for value in precheck_deltas if value <= float(tail_catastrophic_threshold)
                    )
                    precheck_mean_delta = float(precheck_delta_stats.get("mean_delta") or 0.0)
                    precheck_ci_upper = float(precheck_delta_stats.get("ci_upper") or precheck_mean_delta)
                    precheck_reject = (
                        precheck_ci_upper < 0.0
                        or precheck_catastrophic_count >= 2
                        or precheck_mean_delta < -0.75
                    )
                    precheck_trace = {
                        "ran": True,
                        "case_count": len(precheck_cases),
                        "case_ids": [_case_identifier(case) for case in precheck_cases],
                        "delta_stats": precheck_delta_stats,
                        "catastrophic_regression_count": precheck_catastrophic_count,
                        "rejected": bool(precheck_reject),
                        "meta": precheck_meta,
                    }
                    tracker.emit(
                        event_type="precheck_completed",
                        phase="phase_b_precheck_eval",
                        step="precheck_candidate_done",
                        message=f"Epoch {epoch_current}: candidate precheck {'rejected' if precheck_reject else 'passed'}",
                        epoch_current=epoch_current,
                        payload=precheck_trace,
                    )
                    if precheck_reject:
                        changed_keys = []
                        candidate_prompts = dict(current_prompts)
                        tracker.emit(
                            event_type="precheck_rejected_mutations",
                            phase="phase_b_precheck_eval",
                            step="reject_precheck",
                            message=f"Epoch {epoch_current}: candidate rejected by precheck",
                            epoch_current=epoch_current,
                            payload=precheck_trace,
                        )

            generalizer = None
            if (
                changed_keys
                and int(max(0, generalizer_cadence)) > 0
                and (epoch_current % int(max(1, generalizer_cadence)) == 0)
            ):
                tracker.emit(
                    event_type="generalizer_started",
                    phase="generalizer_check",
                    step="run_generalizer",
                    message=f"Epoch {epoch_current}: running generalizer check",
                    epoch_current=epoch_current,
                )
                baseline_generalizer: dict[str, Any] | None = None
                candidate_generalizer: dict[str, Any] | None = None
                if tracker is not None:
                    with tracker.active_call(
                        "generalizer_check",
                        metadata={"epoch": epoch_current},
                    ):
                        baseline_generalizer = generalizer_check(current_prompts, selected)
                        candidate_generalizer = generalizer_check(candidate_prompts, selected)
                else:
                    baseline_generalizer = generalizer_check(current_prompts, selected)
                    candidate_generalizer = generalizer_check(candidate_prompts, selected)
                baseline_generalizer = baseline_generalizer or {}
                candidate_generalizer = candidate_generalizer or {}
                candidate_risk = int(candidate_generalizer.get("overfit_risk_score", 0) or 0)
                baseline_risk = int(baseline_generalizer.get("overfit_risk_score", 0) or 0)
                candidate_suspicious = set(candidate_generalizer.get("suspicious_phrases") or [])
                baseline_suspicious = set(baseline_generalizer.get("suspicious_phrases") or [])
                suspicious_delta = len(candidate_suspicious - baseline_suspicious)
                risk_delta = candidate_risk - baseline_risk
                reject_by_generalizer = (
                    candidate_risk >= GENERALIZER_RISK_REJECT_THRESHOLD
                    and suspicious_delta >= int(max(0, generalizer_suspicious_delta_threshold))
                )
                generalizer = {
                    "baseline": baseline_generalizer,
                    "candidate": candidate_generalizer,
                    "overfit_risk_score": candidate_risk,
                    "suspicious_phrases": sorted(candidate_suspicious),
                    "baseline_risk_score": baseline_risk,
                    "candidate_risk_score": candidate_risk,
                    "risk_delta": risk_delta,
                    "suspicious_delta": suspicious_delta,
                    "reject_by_generalizer": reject_by_generalizer,
                    "reject_rule": (
                        f"candidate_risk >= {GENERALIZER_RISK_REJECT_THRESHOLD} and "
                        f"suspicious_delta >= {int(max(0, generalizer_suspicious_delta_threshold))}"
                    ),
                }
                tracker.emit(
                    event_type="generalizer_completed",
                    phase="generalizer_check",
                    step="run_generalizer_done",
                    message=f"Epoch {epoch_current}: generalizer check complete",
                    epoch_current=epoch_current,
                    payload=generalizer,
                )
                if bool(generalizer.get("reject_by_generalizer")):
                    logger.warning("Generalizer check flagged materially higher overfit risk; rejecting mutations")
                    candidate_prompts = dict(current_prompts)
                    changed_keys = []
                    tracker.emit(
                        event_type="generalizer_rejected_mutations",
                        phase="generalizer_check",
                        step="reject_mutations",
                        message=f"Epoch {epoch_current}: mutations rejected by generalizer",
                        epoch_current=epoch_current,
                        payload=generalizer,
                    )

            candidate_exists = False
            evaluation_mode = "skipped_no_changes"
            truth_state = "synthetic"
            if changed_keys:
                candidate_exists = True
                evaluation_mode = "evaluated"
                truth_state = "complete"
                logger.info(f"Epoch {epoch_current}/{epochs} - Phase B candidate")
                tracker.emit(
                    event_type="phase_started",
                    phase="phase_b_eval",
                    step="evaluate_candidate",
                    message=f"Epoch {epoch_current}: evaluating candidate prompts",
                    epoch_current=epoch_current,
                    payload={"changed_keys": changed_keys},
                )
                if resource_telemetry is not None:
                    with resource_telemetry.phase("phase_b_eval"):
                        results_b, scores_b, phase_b_meta = evaluate_suite(
                            candidate_prompts,
                            selected,
                            thinking_level,
                            phase="phase_b_eval",
                            epoch_current=epoch_current,
                            progress_tracker=tracker,
                            reference_scores=scores_a,
                            early_stop_min_cases=EARLY_STOP_MIN_CASES,
                            early_stop_mean_delta_threshold=EARLY_STOP_MEAN_DELTA_THRESHOLD,
                            eval_mode=eval_mode,
                            eval_min_pairs=train_eval_min_pairs,
                            eval_max_pairs=train_eval_max_pairs,
                            eval_checkpoints=train_eval_checkpoints,
                            bootstrap_resamples=bootstrap_resamples,
                        )
                else:
                    results_b, scores_b, phase_b_meta = evaluate_suite(
                        candidate_prompts,
                        selected,
                        thinking_level,
                        phase="phase_b_eval",
                        epoch_current=epoch_current,
                        progress_tracker=tracker,
                        reference_scores=scores_a,
                        early_stop_min_cases=EARLY_STOP_MIN_CASES,
                        early_stop_mean_delta_threshold=EARLY_STOP_MEAN_DELTA_THRESHOLD,
                        eval_mode=eval_mode,
                        eval_min_pairs=train_eval_min_pairs,
                        eval_max_pairs=train_eval_max_pairs,
                        eval_checkpoints=train_eval_checkpoints,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                avg_score_b = sum(scores_b) / max(len(scores_b), 1)
                stats_b = _score_stats(scores_b)
                phase_b_eval_summary = _summarize_eval_results(results_b)
                phase_b_meta = dict(phase_b_meta or {})
                phase_b_meta["evaluation_mode"] = evaluation_mode
                phase_b_meta["candidate_exists"] = candidate_exists
                phase_b_meta["truth_state"] = truth_state
                tracker.emit(
                    event_type="phase_completed",
                    phase="phase_b_eval",
                    step="evaluate_candidate_done",
                    message=f"Epoch {epoch_current}: candidate evaluation completed",
                    epoch_current=epoch_current,
                    payload={
                        "score_stats": stats_b,
                        "eval_summary": phase_b_eval_summary,
                        "evaluation_meta": phase_b_meta,
                    },
                    metrics={
                        "avg_score_b": avg_score_b,
                        "phase_b_min_score": stats_b["min"],
                        "phase_b_max_score": stats_b["max"],
                        "phase_b_median_score": stats_b["median"],
                        "phase_b_mean_case_time_s": phase_b_eval_summary.get("mean_case_time_s", 0.0),
                        "phase_b_p90_case_time_s": phase_b_eval_summary.get("p90_case_time_s", 0.0),
                        "phase_b_avg_tool_failure_signals": phase_b_eval_summary.get("avg_tool_failure_signals", 0.0),
                        "phase_b_degraded_case_rate": phase_b_eval_summary.get("degraded_case_rate", 0.0),
                        "phase_b_avg_python_exec_attempts": phase_b_eval_summary.get("avg_python_exec_attempts", 0.0),
                        "phase_b_avg_python_exec_failures": phase_b_eval_summary.get("avg_python_exec_failures", 0.0),
                    },
                )
            else:
                results_b, scores_b = results_a, scores_a
                avg_score_b = avg_score_a
                phase_b_eval_summary = dict(phase_a_eval_summary)
                phase_b_meta = {
                    "case_count_requested": len(selected),
                    "case_count_evaluated": len(scores_b),
                    "timeout_enforcement_mode": _timeout_enforcement_mode(TRAINING_CASE_TIME_BUDGET_SECONDS),
                    "early_stop": {"triggered": False},
                    "evaluation_mode": evaluation_mode if ENABLE_NO_CHANGE_STRICT_SEMANTICS else "synthetic",
                    "candidate_exists": False,
                    "truth_state": "synthetic",
                    "synthetic_scores_used": True,
                    "adaptive_eval_trace": {"mode": "synthetic", "decision": "skipped_no_changes", "events": []},
                    "skip_reason": "precheck_rejected"
                    if precheck_trace and bool(precheck_trace.get("rejected"))
                    else "no_candidate_changes",
                }
                tracker.emit(
                    event_type="phase_skipped",
                    phase="phase_b_eval",
                    step="evaluate_candidate_skipped",
                    message=f"Epoch {epoch_current}: candidate evaluation skipped (no changes)",
                    epoch_current=epoch_current,
                    payload={
                        "reason": phase_b_meta.get("skip_reason"),
                        "evaluation_mode": phase_b_meta.get("evaluation_mode"),
                        "candidate_exists": False,
                        "truth_state": "synthetic",
                    },
                )

            update_case_history(
                case_history=case_history,
                selected_cases=selected,
                results_a=results_a,
                results_b=results_b,
                epoch_current=epoch_current,
            )

            winner = "baseline"
            # Exclude infrastructure-degraded pairs from delta stats (Phase 4A)
            pair_count = min(len(results_a), len(results_b))
            clean_scores_a, clean_scores_b = [], []
            excluded_degraded = 0
            for _idx in range(pair_count):
                grade_a = results_a[_idx].get("grade", {}) if _idx < len(results_a) else {}
                grade_b = results_b[_idx].get("grade", {}) if _idx < len(results_b) else {}
                if grade_a.get("infrastructure_degraded") or grade_b.get("infrastructure_degraded"):
                    excluded_degraded += 1
                    continue
                clean_scores_a.append(scores_a[_idx])
                clean_scores_b.append(scores_b[_idx])
            use_clean = len(clean_scores_a) >= 2
            train_delta_stats = _paired_delta_stats(
                clean_scores_a if use_clean else scores_a,
                clean_scores_b if use_clean else scores_b,
                bootstrap_resamples=bootstrap_resamples,
                rng=rng,
            )
            if excluded_degraded > 0:
                train_delta_stats["excluded_degraded_pairs"] = excluded_degraded
                train_delta_stats["used_clean_scores"] = use_clean
            improvement_delta = float(train_delta_stats.get("mean_delta", avg_score_b - avg_score_a))
            prompt_delta_avg = (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0
            paired_eval_case_ids, paired_phase_a_eval_summary, paired_phase_b_eval_summary = _paired_eval_summaries(
                results_a,
                results_b,
            )

            gate_results = _evaluate_candidate_gates(
                delta_stats=train_delta_stats,
                phase_a_summary=paired_phase_a_eval_summary,
                phase_b_summary=paired_phase_b_eval_summary,
                enabled=ENABLE_CANDIDATE_GATES,
                n_pairs=int(train_delta_stats.get("pair_count", 0)),
                quality_gate_min_mean_delta=quality_gate_min_mean_delta,
                quality_gate_min_win_rate=quality_gate_min_win_rate,
                quality_gate_min_p10_delta=quality_gate_min_p10_delta,
                quality_gate_min_worst_case_delta=quality_gate_min_worst_case_delta,
                runtime_gate_mean_ratio_max=runtime_gate_mean_ratio_max,
                runtime_gate_p90_ratio_max=runtime_gate_p90_ratio_max,
                runtime_gate_abs_mean_increase_max_seconds=runtime_gate_abs_mean_increase_max_seconds,
                stability_gate_tool_failure_delta_max=stability_gate_tool_failure_delta_max,
                stability_gate_degraded_rate_delta_max=stability_gate_degraded_rate_delta_max,
                tail_risk_gate_enabled=ENABLE_TAIL_RISK_GATE,
                tail_risk_catastrophic_delta_threshold=tail_catastrophic_threshold,
                tail_risk_max_catastrophic_regressions=tail_max_catastrophic_regressions,
            )
            gate_failure_reasons: list[str] = []
            holdout_case_ids: list[str] = []
            holdout_confirmation: dict[str, Any] | None = None
            holdout_delta_stats: dict[str, Any] | None = None
            holdout_phase_a_summary: dict[str, Any] | None = None
            holdout_phase_b_summary: dict[str, Any] | None = None
            holdout_meta_a: dict[str, Any] | None = None
            holdout_meta_b: dict[str, Any] | None = None
            early_stop_triggered = bool(phase_b_meta.get("early_stop", {}).get("triggered"))
            provisional_candidate = False

            if not changed_keys:
                if precheck_trace and bool(precheck_trace.get("rejected")):
                    gate_failure_reasons.append("precheck_rejected")
                else:
                    gate_failure_reasons.append("no_candidate_changes")
            elif ENABLE_CANDIDATE_GATES:
                provisional_candidate = bool(gate_results.get("all_passed"))
                if not provisional_candidate:
                    gate_failure_reasons.extend(gate_results.get("failed_gates", []))
            else:
                provisional_candidate = avg_score_b > avg_score_a
                if not provisional_candidate:
                    gate_failure_reasons.append("quality_gate")

            if provisional_candidate:
                if holdout_pool:
                    selected_holdout = sample_cases_stratified(
                        sample_state=holdout_sampler_state,
                        sample_size=min(
                            max(int(holdout_sample_size), int(holdout_eval_max_pairs)),
                            len(holdout_pool),
                        ),
                        rng=rng,
                    )
                    holdout_case_ids = [_case_identifier(case) for case in selected_holdout]
                    tracker.emit(
                        event_type="holdout_sample_selected",
                        phase="holdout_eval",
                        step="select_holdout_cases",
                        message=f"Epoch {epoch_current}: selected {len(selected_holdout)} holdout case(s)",
                        epoch_current=epoch_current,
                        payload={
                            "holdout_case_ids": holdout_case_ids,
                            "pool": "holdout",
                            "sample_strategy": "stratified_holdout_rotation",
                        },
                    )
                    if resource_telemetry is not None:
                        with resource_telemetry.phase("holdout_phase_a_eval"):
                            holdout_results_a, holdout_scores_a, holdout_meta_a = evaluate_suite(
                                current_prompts,
                                selected_holdout,
                                thinking_level,
                                phase="holdout_phase_a_eval",
                                epoch_current=epoch_current,
                                progress_tracker=tracker,
                                eval_mode="fixed",
                                eval_min_pairs=holdout_eval_min_pairs,
                                eval_max_pairs=holdout_eval_max_pairs,
                                eval_checkpoints=holdout_eval_checkpoints,
                                bootstrap_resamples=bootstrap_resamples,
                            )
                    else:
                        holdout_results_a, holdout_scores_a, holdout_meta_a = evaluate_suite(
                            current_prompts,
                            selected_holdout,
                            thinking_level,
                            phase="holdout_phase_a_eval",
                            epoch_current=epoch_current,
                            progress_tracker=tracker,
                            eval_mode="fixed",
                            eval_min_pairs=holdout_eval_min_pairs,
                            eval_max_pairs=holdout_eval_max_pairs,
                            eval_checkpoints=holdout_eval_checkpoints,
                            bootstrap_resamples=bootstrap_resamples,
                        )
                    if resource_telemetry is not None:
                        with resource_telemetry.phase("holdout_phase_b_eval"):
                            holdout_results_b, holdout_scores_b, holdout_meta_b = evaluate_suite(
                                candidate_prompts,
                                selected_holdout,
                                thinking_level,
                                phase="holdout_phase_b_eval",
                                epoch_current=epoch_current,
                                progress_tracker=tracker,
                                reference_scores=holdout_scores_a,
                                eval_mode=eval_mode,
                                eval_min_pairs=holdout_eval_min_pairs,
                                eval_max_pairs=holdout_eval_max_pairs,
                                eval_checkpoints=holdout_eval_checkpoints,
                                bootstrap_resamples=bootstrap_resamples,
                            )
                    else:
                        holdout_results_b, holdout_scores_b, holdout_meta_b = evaluate_suite(
                            candidate_prompts,
                            selected_holdout,
                            thinking_level,
                            phase="holdout_phase_b_eval",
                            epoch_current=epoch_current,
                            progress_tracker=tracker,
                            reference_scores=holdout_scores_a,
                            eval_mode=eval_mode,
                            eval_min_pairs=holdout_eval_min_pairs,
                            eval_max_pairs=holdout_eval_max_pairs,
                            eval_checkpoints=holdout_eval_checkpoints,
                            bootstrap_resamples=bootstrap_resamples,
                        )
                    holdout_phase_a_summary = _summarize_eval_results(holdout_results_a)
                    holdout_phase_b_summary = _summarize_eval_results(holdout_results_b)
                    holdout_delta_stats = _paired_delta_stats(
                        holdout_scores_a,
                        holdout_scores_b,
                        bootstrap_resamples=bootstrap_resamples,
                        rng=rng,
                    )
                    holdout_confirmation = _evaluate_holdout_confirmation(
                        holdout_delta_stats,
                        holdout_winrate_min=holdout_winrate_min,
                        holdout_min_mean_delta=-0.10,
                        holdout_catastrophic_threshold=tail_catastrophic_threshold,
                        holdout_max_catastrophic_regressions=tail_max_catastrophic_regressions,
                    )
                    if not holdout_confirmation.get("passed"):
                        gate_failure_reasons.append("holdout_confirmation_gate")
                else:
                    holdout_confirmation = {
                        "passed": True,
                        "wins": 0,
                        "pairs": 0,
                        "win_rate": 0.0,
                        "win_rate_min": float(holdout_winrate_min),
                        "mean_delta": 0.0,
                        "mean_delta_min": -0.10,
                        "ci_lower": 0.0,
                        "ci_upper": 0.0,
                        "catastrophic_threshold": float(tail_catastrophic_threshold),
                        "max_catastrophic_regressions": int(max(0, tail_max_catastrophic_regressions)),
                        "catastrophic_regression_count": 0,
                        "checks": {
                            "win_rate_minimum": True,
                            "mean_delta_floor": True,
                            "catastrophic_limit": True,
                        },
                        "rule": "holdout unavailable -> pass",
                    }

                if holdout_confirmation and holdout_confirmation.get("passed"):
                    winner = "candidate"

            rejection_reason_codes = _build_rejection_reason_codes(
                gate_failure_reasons=gate_failure_reasons,
                gate_results=gate_results,
                candidate_exists=bool(candidate_exists),
                evaluation_mode=str(evaluation_mode),
            )
            decision_why = {
                "winner": winner,
                "failed_checks": list(gate_failure_reasons),
                "rejection_reason_codes": rejection_reason_codes,
                "candidate_exists": bool(candidate_exists),
                "evaluation_mode": str(evaluation_mode),
                "truth_state": str(truth_state),
                "resource_costs": {
                    "phase_a": phase_a_eval_summary,
                    "phase_b": phase_b_eval_summary,
                    "holdout_phase_a": holdout_phase_a_summary,
                    "holdout_phase_b": holdout_phase_b_summary,
                },
            }

            if winner == "candidate":
                generation += 1
                current_prompts = candidate_prompts
                for bid in changed_keys:
                    if bid not in mutation_coverage:
                        mutation_coverage[bid] = {"targeted": 0, "mutations_generated": 0, "validation_passed": 0, "gates_passed": 0}
                    mutation_coverage[bid]["gates_passed"] += 1

            selection_slot_by_case_id = {
                str(item.get("case_id")): str(item.get("slot") or "unknown")
                for item in selection_diagnostics
                if isinstance(item, dict) and item.get("case_id")
            }
            selection_slot_deltas: dict[str, list[float]] = {}
            for idx, case in enumerate(selected[: min(len(scores_a), len(scores_b))]):
                case_id = _case_identifier(case)
                slot = selection_slot_by_case_id.get(case_id, "unknown")
                selection_slot_deltas.setdefault(slot, []).append(float(scores_b[idx]) - float(scores_a[idx]))

            def _improvement_rate(values: list[float]) -> float | None:
                if not values:
                    return None
                return float(sum(1 for value in values if value > 0.0) / len(values))

            replay_improvement_rate = _improvement_rate(selection_slot_deltas.get("replay", []))
            coverage_improvement_rate = _improvement_rate(selection_slot_deltas.get("coverage", []))
            selection_effectiveness = {
                "slot_counts": {slot: len(values) for slot, values in sorted(selection_slot_deltas.items())},
                "slot_mean_delta": {
                    slot: (float(sum(values) / len(values)) if values else 0.0)
                    for slot, values in sorted(selection_slot_deltas.items())
                },
                "replay_improvement_rate": replay_improvement_rate,
                "coverage_improvement_rate": coverage_improvement_rate,
                "exploration_improvement_rate": _improvement_rate(selection_slot_deltas.get("exploration", [])),
                "replay_vs_coverage_delta": (
                    float(replay_improvement_rate - coverage_improvement_rate)
                    if replay_improvement_rate is not None and coverage_improvement_rate is not None
                    else None
                ),
            }

            selection_stats = {
                "train_delta_stats": train_delta_stats,
                "holdout_delta_stats": holdout_delta_stats,
                "holdout_confirmation": holdout_confirmation,
                "early_stop_triggered": early_stop_triggered,
                "prompt_delta_avg_total": prompt_delta_avg,
                "paired_eval_case_ids": paired_eval_case_ids,
                "selection_mode": str(selection_mode),
                "selection_diagnostics": selection_diagnostics,
                "rca_selection_diagnostics": rca_selection_diagnostics,
                "selection_effectiveness": selection_effectiveness,
                "holdout_winrate": (holdout_confirmation or {}).get("win_rate"),
                "candidate_exists": bool(candidate_exists),
                "evaluation_mode": str(evaluation_mode),
                "truth_state": str(truth_state),
                "precheck_trace": precheck_trace,
            }
            tracker.emit(
                event_type="candidate_gate_decision",
                phase="selection",
                step="apply_candidate_gates",
                message=f"Epoch {epoch_current}: candidate gate decision = {winner}",
                epoch_current=epoch_current,
                payload={
                    "winner": winner,
                    "changed_keys_count": len(changed_keys),
                    "prompt_delta_avg_total": prompt_delta_avg,
                    "gates_enabled": ENABLE_CANDIDATE_GATES,
                    "gate_results": gate_results,
                    "gate_failure_reasons": gate_failure_reasons,
                    "rejection_reason_codes": rejection_reason_codes,
                    "phase_a_eval_summary": phase_a_eval_summary,
                    "phase_b_eval_summary": phase_b_eval_summary,
                    "paired_phase_a_eval_summary": paired_phase_a_eval_summary,
                    "paired_phase_b_eval_summary": paired_phase_b_eval_summary,
                    "paired_eval_case_ids": paired_eval_case_ids,
                    "selection_stats": selection_stats,
                    "candidate_exists": bool(candidate_exists),
                    "evaluation_mode": str(evaluation_mode),
                    "truth_state": str(truth_state),
                    "decision_why": decision_why,
                },
                metrics={
                    "candidate_gates_enabled": 1 if ENABLE_CANDIDATE_GATES else 0,
                    "candidate_gate_passed": 1 if winner == "candidate" else 0,
                    "candidate_exists": 1 if candidate_exists else 0,
                    "candidate_runtime_mean_ratio": gate_results.get("ratios", {}).get("mean_case_time_ratio") or 0.0,
                    "candidate_runtime_p90_ratio": gate_results.get("ratios", {}).get("p90_case_time_ratio") or 0.0,
                    "candidate_prompt_delta_avg_total": prompt_delta_avg,
                },
            )

            mutation_attempts = [
                {
                    "block_id": item.get("block_id"),
                    "attempts": item.get("mutation_attempts", []),
                }
                for item in prompt_scoring
                if isinstance(item, dict) and item.get("mutation_attempts")
            ]
            retried_block_count = sum(
                1
                for item in mutation_attempts
                if isinstance(item.get("attempts"), list) and len(item.get("attempts", [])) > 1
            )
            total_retry_attempts = sum(
                max(0, len(item.get("attempts", [])) - 1)
                for item in mutation_attempts
                if isinstance(item.get("attempts"), list)
            )
            mutation_retry_summary = {
                "enabled": bool(mutation_retry_enabled and ENABLE_MUTATION_RETRY_V2),
                "max_retries": int(max(0, mutation_max_retries)),
                "blocks_with_attempts": len(mutation_attempts),
                "retried_blocks": retried_block_count,
                "total_retry_attempts": total_retry_attempts,
            }
            timeout_stats = {
                "phase_a": _timeout_stats(results_a),
                "phase_b": _timeout_stats(results_b),
                "holdout_phase_a": _timeout_stats(holdout_results_a) if holdout_phase_a_summary is not None else None,
                "holdout_phase_b": _timeout_stats(holdout_results_b) if holdout_phase_b_summary is not None else None,
            }
            _update_block_impact_history(
                block_impact_history=block_impact_history,
                mutated_block_ids=list(changed_keys),
                winner=winner,
                train_delta_stats=train_delta_stats,
            )

            resource_profile = {
                "llm_calls": 0,
                "llm_tokens_in": 0,
                "llm_tokens_out": 0,
                "phase_wall_seconds": {},
                "pipeline_seconds": float(phase_a_eval_summary.get("sum_pipeline_run_s", 0.0) or 0.0)
                + float(phase_b_eval_summary.get("sum_pipeline_run_s", 0.0) or 0.0),
                "grading_seconds": float(phase_a_eval_summary.get("sum_grading_s", 0.0) or 0.0)
                + float(phase_b_eval_summary.get("sum_grading_s", 0.0) or 0.0),
                "rca_seconds": 0.0,
                "mutation_seconds": 0.0,
            }
            if holdout_phase_a_summary is not None:
                resource_profile["pipeline_seconds"] += float(
                    holdout_phase_a_summary.get("sum_pipeline_run_s", 0.0) or 0.0
                )
                resource_profile["grading_seconds"] += float(
                    holdout_phase_a_summary.get("sum_grading_s", 0.0) or 0.0
                )
            if holdout_phase_b_summary is not None:
                resource_profile["pipeline_seconds"] += float(
                    holdout_phase_b_summary.get("sum_pipeline_run_s", 0.0) or 0.0
                )
                resource_profile["grading_seconds"] += float(
                    holdout_phase_b_summary.get("sum_grading_s", 0.0) or 0.0
                )

            if resource_telemetry is not None:
                resource_delta = resource_telemetry.delta_since(resource_baseline_snapshot)
                resource_baseline_snapshot = resource_telemetry.snapshot()
                resource_profile["llm_calls"] = int(resource_delta.get("llm_calls", 0) or 0)
                resource_profile["llm_tokens_in"] = int(resource_delta.get("llm_tokens_in", 0) or 0)
                resource_profile["llm_tokens_out"] = int(resource_delta.get("llm_tokens_out", 0) or 0)
                resource_profile["phase_wall_seconds"] = dict(resource_delta.get("phase_wall_seconds") or {})
                phase_wall = resource_profile["phase_wall_seconds"]
                resource_profile["rca_seconds"] = float(
                    (phase_wall.get("rca", 0.0) or 0.0)
                    + (phase_wall.get("rca_high_level", 0.0) or 0.0)
                )
                resource_profile["mutation_seconds"] = float(
                    (phase_wall.get("mutation_stage_a", 0.0) or 0.0)
                    + (phase_wall.get("mutation_stage_b", 0.0) or 0.0)
                    + (phase_wall.get("prompt_improvement", 0.0) or 0.0)
                )
            total_tokens = int(resource_profile.get("llm_tokens_in", 0) or 0) + int(
                resource_profile.get("llm_tokens_out", 0) or 0
            )
            total_wall_seconds = sum(
                float(value or 0.0)
                for value in (resource_profile.get("phase_wall_seconds") or {}).values()
            )
            improvement_per_1k_tokens = (
                float(improvement_delta) / (float(total_tokens) / 1000.0) if total_tokens > 0 else None
            )
            improvement_per_phase_second = (
                float(improvement_delta) / float(total_wall_seconds) if total_wall_seconds > 0 else None
            )

            data_quality_flags = ["complete"]
            if not candidate_exists:
                data_quality_flags = ["synthetic"]

            metadata = {
                "epoch": epoch,
                "epoch_index": epoch,
                "epoch_number": epoch_current,
                "run_id": run_id,
                "avg_score_a": avg_score_a,
                "avg_score_b": avg_score_b,
                "improvement_delta": improvement_delta,
                "mean_delta": train_delta_stats.get("mean_delta"),
                "delta_ci": {
                    "ci_lower": train_delta_stats.get("ci_lower"),
                    "ci_upper": train_delta_stats.get("ci_upper"),
                },
                "winner": winner,
                "changed_keys": changed_keys if winner == "candidate" else [],
                "mutated_block_ids": changed_keys,
                "candidate_exists": bool(candidate_exists),
                "evaluation_mode": str(evaluation_mode),
                "truth_state": str(truth_state),
                "legacy_inferred": False,
                "data_quality_flags": data_quality_flags,
                "scores_a": scores_a,
                "scores_b": scores_b,
                "thinking_level": thinking_level,
                "num_failures_in_a": len(failed),
                "num_rca_items": len(all_analyses),
                "generalizer": generalizer,
                "phase_a_eval_summary": phase_a_eval_summary,
                "phase_b_eval_summary": phase_b_eval_summary,
                "paired_phase_a_eval_summary": paired_phase_a_eval_summary,
                "paired_phase_b_eval_summary": paired_phase_b_eval_summary,
                "paired_eval_case_ids": paired_eval_case_ids,
                "phase_a_eval_meta": phase_a_meta,
                "phase_b_eval_meta": phase_b_meta,
                "adaptive_eval_trace": {
                    "train_phase_b": (phase_b_meta or {}).get("adaptive_eval_trace"),
                    "holdout_phase_b": (holdout_meta_b or {}).get("adaptive_eval_trace") if holdout_meta_b else None,
                },
                "provisional_decision_trace": {
                    "provisional_candidate": bool(provisional_candidate),
                    "gate_failure_reasons": list(gate_failure_reasons),
                    "gate_results": gate_results,
                    "evaluation_mode": str(evaluation_mode),
                },
                "holdout_decision_trace": {
                    "holdout_case_ids": holdout_case_ids,
                    "holdout_confirmation": holdout_confirmation,
                    "holdout_delta_stats": holdout_delta_stats,
                    "holdout_phase_a_meta": holdout_meta_a,
                    "holdout_phase_b_meta": holdout_meta_b,
                },
                "holdout_phase_a_eval_summary": holdout_phase_a_summary,
                "holdout_phase_b_eval_summary": holdout_phase_b_summary,
                "holdout_phase_a_eval_meta": holdout_meta_a,
                "holdout_phase_b_eval_meta": holdout_meta_b,
                "candidate_gate_results": gate_results,
                "candidate_gate_failure_reasons": gate_failure_reasons,
                "rejection_reason_codes": rejection_reason_codes,
                "selection_blocked_reason": (
                    "precheck_rejected"
                    if (not candidate_exists and precheck_trace and bool(precheck_trace.get("rejected")))
                    else ("no_candidate_changes" if not candidate_exists else None)
                ),
                "decision_why": decision_why,
                "prompt_scoring": prompt_scoring,
                "mutation_rejection_breakdown": rejection_breakdown,
                "mutation_rejection_issue_matrix": rejection_issue_matrix,
                "sample_strategy": str(selection_mode),
                "train_case_ids": [_case_identifier(case) for case in selected],
                "holdout_case_ids": holdout_case_ids,
                "rca_case_ids": rca_case_ids,
                "selection_stats": selection_stats,
                "selection_effectiveness": selection_effectiveness,
                "selection_diagnostics": selection_diagnostics,
                "rca_high_level_summary": high_level_rca,
                "precheck_trace": precheck_trace,
                "mutation_accept_threshold": int(max(1, mutation_accept_threshold)),
                "holdout_winrate": (holdout_confirmation or {}).get("win_rate"),
                "resource_profile": resource_profile,
                "improvement_per_1k_tokens": improvement_per_1k_tokens,
                "improvement_per_phase_second": improvement_per_phase_second,
                "evaluation_layers": {
                    "heuristic_mutation_scorer": {
                        "name": "prompt_criteria_scorer",
                        "role": "search_ranking_only",
                        "authoritative_for_promotion": False,
                    },
                    "end_to_end_train": {
                        "name": "paired_case_evaluation_train",
                        "role": "provisional_promotion_truth",
                        "authoritative_for_promotion": True,
                    },
                    "end_to_end_holdout": {
                        "name": "paired_case_evaluation_holdout",
                        "role": "final_confirmation",
                        "authoritative_for_promotion": True,
                    },
                },
                "gate_rule_versions": {
                    "quality": "multi_constraint_quality_v3",
                    "runtime": "balanced_lenient_v2",
                    "stability": "balanced_lenient_v2",
                    "tail_risk": "catastrophic_guardrail_v1",
                    "holdout": "winrate_v2",
                },
                "runtime_gate_profile": {
                    "mean_ratio_max": float(runtime_gate_mean_ratio_max),
                    "p90_ratio_max": float(runtime_gate_p90_ratio_max),
                    "abs_mean_increase_max_seconds": float(runtime_gate_abs_mean_increase_max_seconds),
                },
                "stability_gate_profile": {
                    "tool_failure_delta_max": float(stability_gate_tool_failure_delta_max),
                    "degraded_rate_delta_max": float(stability_gate_degraded_rate_delta_max),
                },
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "train_split": split_metadata["train_split"],
                "holdout_split": split_metadata["holdout_split"],
                "early_stop": phase_b_meta.get("early_stop", {"triggered": False}),
                "events_archive_path": events_archive_path,
                "mutation_retry_summary": mutation_retry_summary,
                "mutation_attempts": mutation_attempts,
                "timeout_stats": timeout_stats,
                "case_history_snapshot": case_history,
                "block_impact_history_snapshot": block_impact_history,
                "prompt_scoring_summary": {
                    "changed_blocks": len(changed_blocks),
                    "scored_blocks": len(scored_blocks),
                    "accepted_blocks": len(accepted_blocks),
                    "rejected_blocks": len(rejected_blocks),
                    "contract_rejected_blocks": len(contract_rejected_blocks),
                    "score_rejected_blocks": len(score_rejected_blocks),
                    "baseline_avg_total": (sum(baseline_totals) / len(baseline_totals)) if baseline_totals else 0.0,
                    "candidate_avg_total": (sum(candidate_totals) / len(candidate_totals)) if candidate_totals else 0.0,
                    "delta_avg_total": (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0,
                },
                "mutation_style_summary": mutation_style_summary,
                "mutation_staging": {
                    "enabled": bool(ENABLE_MUTATION_BUDGET_OPT_V1),
                    "stage_a_top_k": int(max(1, mutation_stage_a_top_k)),
                    "stage_a_target_ids": stage_a_ids,
                    "stage_b_target_ids": stage_b_ids,
                    "stage_b_ran": bool(ran_stage_b),
                },
                "resource_target_reduction": float(max(0.0, resource_target_reduction)),
                "mutation_stage_a_top_k": int(max(1, mutation_stage_a_top_k)),
                "mutation_precheck_cases": int(max(0, mutation_precheck_cases)),
                "eval_mode": str(eval_mode),
            }
            store.save_generation(current_prompts, generation, metadata)
            tracker.emit(
                event_type="mutation_coverage_summary",
                phase="prompt_improvement",
                step="coverage_summary",
                message=f"Epoch {epoch_current}: mutation coverage summary",
                epoch_current=epoch_current,
                payload={"mutation_coverage": dict(mutation_coverage)},
            )
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
                    "avg_score_a": avg_score_a,
                    "avg_score_b": avg_score_b,
                    "improvement_delta": improvement_delta,
                    "candidate_gate_failure_reasons": gate_failure_reasons,
                    "candidate_gate_results": gate_results,
                    "selection_stats": selection_stats,
                },
                metrics={
                    "avg_score_a": avg_score_a,
                    "avg_score_b": avg_score_b,
                    "improvement_delta": improvement_delta,
                    "generation": generation,
                    "num_failures_in_a": len(failed),
                    "num_rca_items": len(all_analyses),
                    "candidate_gates_enabled": 1 if ENABLE_CANDIDATE_GATES else 0,
                    "candidate_gate_passed": 1 if winner == "candidate" else 0,
                    "phase_a_mean_case_time_s": phase_a_eval_summary.get("mean_case_time_s", 0.0),
                    "phase_b_mean_case_time_s": phase_b_eval_summary.get("mean_case_time_s", 0.0),
                    "phase_a_p90_case_time_s": phase_a_eval_summary.get("p90_case_time_s", 0.0),
                    "phase_b_p90_case_time_s": phase_b_eval_summary.get("p90_case_time_s", 0.0),
                    "phase_a_avg_tool_failure_signals": phase_a_eval_summary.get("avg_tool_failure_signals", 0.0),
                    "phase_b_avg_tool_failure_signals": phase_b_eval_summary.get("avg_tool_failure_signals", 0.0),
                    "phase_a_degraded_case_rate": phase_a_eval_summary.get("degraded_case_rate", 0.0),
                    "phase_b_degraded_case_rate": phase_b_eval_summary.get("degraded_case_rate", 0.0),
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
            "events_archive_path": events_archive_path,
            "sample_strategy": str(selection_mode),
            "train_split_size": len(train_pool),
            "holdout_split_size": len(holdout_pool),
        }
        tracker.complete(metrics={"generation": generation, "history_entries": result["history_entries"]})
        return result
    except Exception as exc:
        tracker.fail(str(exc))
        logger.exception(f"Training run failed: {exc}")
        raise
    finally:
        _ACTIVE_RESOURCE_TELEMETRY = previous_resource_telemetry


def run_training_loop(
    base_prompts_path: str = "data/prompts.json",
    output_path: str = "data/prompt_suite_generations.json",
    test_cases_path: str = "data/prompt_train_cases.json",
    epochs: int = 10,
    num_test_cases_per_trial: int = 10,
    holdout_sample_size: int = 6,
    rca_case_budget: int = 3,
    mutation_block_budget: int = 1,
    mutation_accept_threshold: int = MUTATION_ACCEPTANCE_TOTAL_MIN,
    mutation_tournament_size: int = DEFAULT_MUTATION_TOURNAMENT_SIZE,
    mutation_retry_enabled: bool = True,
    mutation_max_retries: int = DEFAULT_MUTATION_MAX_RETRIES,
    generalizer_cadence: int = 3,
    generalizer_suspicious_delta_threshold: int = GENERALIZER_SUSPICIOUS_DELTA_THRESHOLD,
    bootstrap_resamples: int = 1000,
    rca_threshold: float = DEFAULT_RCA_THRESHOLD,
    rca_fallback_fraction: float = DEFAULT_RCA_FALLBACK_FRACTION,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    runtime_gate_mean_ratio_max: float = RUNTIME_GATE_MEAN_MAX_RATIO,
    runtime_gate_p90_ratio_max: float = RUNTIME_GATE_P90_MAX_RATIO,
    runtime_gate_abs_mean_increase_max_seconds: float = RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS,
    stability_gate_tool_failure_delta_max: float = STABILITY_GATE_MAX_TOOL_FAILURE_DELTA,
    stability_gate_degraded_rate_delta_max: float = DEGRADATION_GATE_MAX_DELTA,
    holdout_winrate_min: float = HOLDOUT_GATE_MIN_WIN_RATE,
    quality_gate_min_mean_delta: float = QUALITY_GATE_MIN_MEAN_DELTA,
    quality_gate_min_win_rate: float = QUALITY_GATE_MIN_WIN_RATE,
    quality_gate_min_p10_delta: float = QUALITY_GATE_MIN_P10_DELTA,
    quality_gate_min_worst_case_delta: float = QUALITY_GATE_MIN_WORST_CASE_DELTA,
    tail_catastrophic_threshold: float = TAIL_RISK_CATASTROPHIC_DELTA_THRESHOLD,
    tail_max_catastrophic_regressions: int = TAIL_RISK_MAX_CATASTROPHIC_REGRESSIONS,
    eval_mode: str = DEFAULT_EVAL_MODE,
    train_eval_min_pairs: int = DEFAULT_TRAIN_EVAL_MIN_PAIRS,
    train_eval_max_pairs: int = DEFAULT_TRAIN_EVAL_MAX_PAIRS,
    train_eval_checkpoints: list[int] | tuple[int, ...] | str | None = None,
    holdout_eval_min_pairs: int = DEFAULT_HOLDOUT_EVAL_MIN_PAIRS,
    holdout_eval_max_pairs: int = DEFAULT_HOLDOUT_EVAL_MAX_PAIRS,
    holdout_eval_checkpoints: list[int] | tuple[int, ...] | str | None = None,
    mutation_stage_a_top_k: int = DEFAULT_MUTATION_STAGE_A_TOP_K,
    mutation_precheck_cases: int = DEFAULT_MUTATION_PRECHECK_CASES,
    resource_target_reduction: float = DEFAULT_RESOURCE_TARGET_REDUCTION,
    holdout_split_ratio: float = DEFAULT_HOLDOUT_SPLIT_RATIO,
    random_seed: int = 42,
    thinking_level: Literal["low", "med-synth", "med-plan", "high"] = "med-synth",
    progress_status_path: str = "data/training_status.json",
    progress_events_path: str = "data/training_events.jsonl",
    progress_events_archive_dir: str = "data/training_events",
    track_progress: bool = True,
) -> dict[str, Any]:
    return train_ab_loop(
        base_prompts_path=base_prompts_path,
        output_path=output_path,
        test_cases_path=test_cases_path,
        epochs=epochs,
        num_test_cases_per_trial=num_test_cases_per_trial,
        holdout_sample_size=holdout_sample_size,
        rca_case_budget=rca_case_budget,
        mutation_block_budget=mutation_block_budget,
        mutation_accept_threshold=mutation_accept_threshold,
        mutation_tournament_size=mutation_tournament_size,
        mutation_retry_enabled=mutation_retry_enabled,
        mutation_max_retries=mutation_max_retries,
        generalizer_cadence=generalizer_cadence,
        generalizer_suspicious_delta_threshold=generalizer_suspicious_delta_threshold,
        bootstrap_resamples=bootstrap_resamples,
        rca_threshold=rca_threshold,
        rca_fallback_fraction=rca_fallback_fraction,
        selection_mode=selection_mode,
        runtime_gate_mean_ratio_max=runtime_gate_mean_ratio_max,
        runtime_gate_p90_ratio_max=runtime_gate_p90_ratio_max,
        runtime_gate_abs_mean_increase_max_seconds=runtime_gate_abs_mean_increase_max_seconds,
        stability_gate_tool_failure_delta_max=stability_gate_tool_failure_delta_max,
        stability_gate_degraded_rate_delta_max=stability_gate_degraded_rate_delta_max,
        holdout_winrate_min=holdout_winrate_min,
        quality_gate_min_mean_delta=quality_gate_min_mean_delta,
        quality_gate_min_win_rate=quality_gate_min_win_rate,
        quality_gate_min_p10_delta=quality_gate_min_p10_delta,
        quality_gate_min_worst_case_delta=quality_gate_min_worst_case_delta,
        tail_catastrophic_threshold=tail_catastrophic_threshold,
        tail_max_catastrophic_regressions=tail_max_catastrophic_regressions,
        eval_mode=eval_mode,
        train_eval_min_pairs=train_eval_min_pairs,
        train_eval_max_pairs=train_eval_max_pairs,
        train_eval_checkpoints=train_eval_checkpoints,
        holdout_eval_min_pairs=holdout_eval_min_pairs,
        holdout_eval_max_pairs=holdout_eval_max_pairs,
        holdout_eval_checkpoints=holdout_eval_checkpoints,
        mutation_stage_a_top_k=mutation_stage_a_top_k,
        mutation_precheck_cases=mutation_precheck_cases,
        resource_target_reduction=resource_target_reduction,
        holdout_split_ratio=holdout_split_ratio,
        random_seed=random_seed,
        thinking_level=thinking_level,
        progress_status_path=progress_status_path,
        progress_events_path=progress_events_path,
        progress_events_archive_dir=progress_events_archive_dir,
        track_progress=track_progress,
    )


if __name__ == "__main__":
    result = run_training_loop()
    print(json.dumps(result, indent=2))
