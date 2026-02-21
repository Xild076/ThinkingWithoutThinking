from __future__ import annotations

import copy
import json
import logging
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
TRAINING_CASE_TIME_BUDGET_SECONDS = _env_float("TRAINING_CASE_TIME_BUDGET_SECONDS", 600.0)
STALL_WARNING_THRESHOLD_SECONDS = _env_float("TRAINING_STALL_WARNING_SECONDS", 600.0)

RUNTIME_GATE_MEAN_MAX_RATIO = 1.20
RUNTIME_GATE_P90_MAX_RATIO = 1.30
RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS = 45.0
STABILITY_GATE_MAX_TOOL_FAILURE_DELTA = 0.2
DEGRADATION_GATE_MAX_DELTA = 0.05
PROMPT_SCORE_ACCEPTANCE_RATIO = 1.05
QUALITY_GATE_MIN_MEAN_DELTA = 0.10
QUALITY_GATE_MIN_CI_LOWER = -0.05
HOLDOUT_GATE_MIN_MEAN_DELTA = 0.0
HOLDOUT_GATE_MIN_CI_LOWER = -0.10
EARLY_STOP_MIN_CASES = 4
EARLY_STOP_MEAN_DELTA_THRESHOLD = -0.35
DEFAULT_HOLDOUT_SPLIT_RATIO = 0.2

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


def _should_skip_oss_for_prompt(prompt: str, token_budget: int = 5500) -> bool:
    return _estimated_tokens_from_text(prompt) > token_budget


def _extract_explicit_json_keys(prompt_text: str) -> set[str]:
    return set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', prompt_text or ""))


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
    "python_tool_prompt_missing_repair_or_schema_markers",
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
        "avg_tool_failure_signals": _mean(tool_failure_counts) or 0.0,
        "degraded_case_rate": degraded_rate,
        "degraded_case_count": degraded_count,
        "avg_python_exec_attempts": _mean(python_attempts) or 0.0,
        "avg_python_exec_failures": _mean(python_failures) or 0.0,
        "tool_invocation_totals": tool_invocation_totals,
        "tool_error_totals": tool_error_totals,
    }


def _evaluate_candidate_gates(
    *,
    delta_stats: dict[str, Any],
    phase_a_summary: dict[str, Any],
    phase_b_summary: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    mean_delta = float(delta_stats.get("mean_delta", 0.0) or 0.0)
    ci_lower = float(delta_stats.get("ci_lower", mean_delta) or mean_delta)
    ci_upper = float(delta_stats.get("ci_upper", mean_delta) or mean_delta)

    quality_gate = {
        "name": "quality_gate",
        "passed": mean_delta >= QUALITY_GATE_MIN_MEAN_DELTA and ci_lower > QUALITY_GATE_MIN_CI_LOWER,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rule": f"mean_delta >= {QUALITY_GATE_MIN_MEAN_DELTA} and ci_lower > {QUALITY_GATE_MIN_CI_LOWER}",
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
        "passed": (mean_ratio is None or mean_ratio <= RUNTIME_GATE_MEAN_MAX_RATIO)
        and (p90_ratio is None or p90_ratio <= RUNTIME_GATE_P90_MAX_RATIO),
        "baseline_mean_case_time_s": mean_a,
        "candidate_mean_case_time_s": mean_b,
        "baseline_p90_case_time_s": p90_a,
        "candidate_p90_case_time_s": p90_b,
        "mean_ratio_b_over_a": mean_ratio,
        "p90_ratio_b_over_a": p90_ratio,
        "mean_delta_seconds_b_minus_a": mean_delta_seconds,
        "passed_abs_mean_delta": mean_delta_seconds <= RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS,
        "rule": (
            f"mean_ratio <= {RUNTIME_GATE_MEAN_MAX_RATIO}, "
            f"p90_ratio <= {RUNTIME_GATE_P90_MAX_RATIO}, "
            f"and mean_delta_seconds <= {RUNTIME_GATE_MAX_ABS_MEAN_INCREASE_SECONDS}"
        ),
    }
    runtime_gate["passed"] = bool(runtime_gate["passed"]) and bool(runtime_gate["passed_abs_mean_delta"])

    tool_failure_delta = failure_b - failure_a
    degraded_rate_delta = degraded_rate_b - degraded_rate_a

    stability_gate = {
        "name": "stability_gate",
        "passed": tool_failure_delta <= STABILITY_GATE_MAX_TOOL_FAILURE_DELTA
        and degraded_rate_delta <= DEGRADATION_GATE_MAX_DELTA,
        "baseline_avg_tool_failure_signals": failure_a,
        "candidate_avg_tool_failure_signals": failure_b,
        "tool_failure_delta_b_minus_a": tool_failure_delta,
        "baseline_degraded_case_rate": degraded_rate_a,
        "candidate_degraded_case_rate": degraded_rate_b,
        "degraded_rate_delta_b_minus_a": degraded_rate_delta,
        "rule": (
            f"tool_failure_delta <= {STABILITY_GATE_MAX_TOOL_FAILURE_DELTA} "
            f"and degraded_rate_delta <= {DEGRADATION_GATE_MAX_DELTA}"
        ),
    }

    gates = [quality_gate, runtime_gate, stability_gate]
    all_passed = all(bool(gate.get("passed")) for gate in gates)
    failed_gates = [str(gate.get("name")) for gate in gates if not bool(gate.get("passed"))]

    return {
        "enabled": bool(enabled),
        "all_passed": bool(all_passed),
        "failed_gates": failed_gates,
        "gates": {gate["name"]: gate for gate in gates},
        "ratios": {
            "mean_case_time_ratio": mean_ratio,
            "p90_case_time_ratio": p90_ratio,
        },
        "delta_stats": {
            "mean_delta": mean_delta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "deltas": {
            "tool_failure_delta": tool_failure_delta,
            "degraded_rate_delta": degraded_rate_delta,
            "mean_case_time_delta_seconds": mean_delta_seconds,
        },
    }


def _evaluate_holdout_confirmation(delta_stats: dict[str, Any]) -> dict[str, Any]:
    mean_delta = float(delta_stats.get("mean_delta", 0.0) or 0.0)
    ci_lower = float(delta_stats.get("ci_lower", mean_delta) or mean_delta)
    ci_upper = float(delta_stats.get("ci_upper", mean_delta) or mean_delta)
    passed = mean_delta >= HOLDOUT_GATE_MIN_MEAN_DELTA and ci_lower > HOLDOUT_GATE_MIN_CI_LOWER
    return {
        "passed": passed,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rule": f"mean_delta >= {HOLDOUT_GATE_MIN_MEAN_DELTA} and ci_lower > {HOLDOUT_GATE_MIN_CI_LOWER}",
    }


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
    attempts: list[tuple[str, BaseModel]] = []
    model_order: list[str] = ["oss120b", "nemotron"]
    if _should_skip_oss_for_prompt(prompt):
        logger.info("Skipping oss120b for oversized prompt; using nemotron directly")
        model_order = ["nemotron"]

    for model in model_order:
        retries = 2 if model == "oss120b" else 5
        max_retry_wait = 8.0 if model == "oss120b" else 30.0
        output_raw = generate_text(
            prompt=prompt,
            model=model,
            schema=schema,
            temperature=temperature,
            retries=retries,
            max_total_retry_wait=max_retry_wait,
        )
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
        self._last_case_activity_unix: float | None = None
        self._last_stalled_warning_unix: float = 0.0
        self.stalled_warning_threshold_seconds = max(60.0, float(STALL_WARNING_THRESHOLD_SECONDS))
        self._run_started_perf = time.perf_counter()
        self._phase_started_perf = self._run_started_perf
        self._last_phase = "init"
        self.status_path = Path(status_path)
        self.events_path = Path(events_path)
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
        timestamp = _iso_now()
        timestamp_unix = round(time.time(), 3)
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
            "payload": payload or {},
            "current_case": current_case,
            "metrics": metrics or {},
            "active_call": self._active_call_snapshot(timestamp_unix),
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

        self._write_status()
        self._append_event(event)
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

    graded = generate_text(
        prompt=grading_prompt,
        model="nemotron",
        schema=GenericGradeSchema,
        temperature=0.0,
        retries=5,
        max_total_retry_wait=30.0,
    )

    payload = graded.model_dump()
    payload["aggregate_score"] = _weighted_aggregate(payload)
    return payload


def root_cause_analysis(
    prompt: str,
    run_output: dict[str, Any],
    major_issues: str,
    valid_blocks: list[str],
    block_details_map: dict[str, dict[str, Any]],
    criteria_context_text: str | None = None,
    case_context: dict[str, Any] | None = None,
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
    run_output_json = _safe_json_dumps(compact_run_output, max_chars=14000)
    case_context_text = _safe_json_dumps(case_context or {}, max_chars=1200)

    rca_prompt = (
        "You are an expert pipeline RCA analyst.\n"
        "Use BOTH generic quality failures and block-specific prompt criteria misalignment.\n"
        "You MUST only reference valid block IDs.\n\n"
        f"Valid blocks:\n{', '.join(valid_blocks)}\n\n"
        f"Case context:\n{case_context_text}\n\n"
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


def _select_rca_cases(results: list[dict[str, Any]], rca_case_budget: int) -> list[dict[str, Any]]:
    budget = max(0, int(rca_case_budget))
    if budget == 0:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _append_if_new(item: dict[str, Any]) -> None:
        case_payload = item.get("case")
        if isinstance(case_payload, dict):
            case_id = _case_identifier(case_payload)
        else:
            case_id = str(item.get("prompt", ""))[:220]
        if case_id in selected_ids:
            return
        selected_ids.add(case_id)
        selected.append(item)

    degraded_or_timeout = [
        result
        for result in results
        if bool(result.get("case_stats", {}).get("timed_out"))
        or bool(result.get("case_stats", {}).get("degraded_mode_active"))
    ]
    degraded_or_timeout.sort(
        key=lambda item: float(item.get("grade", {}).get("aggregate_score", 0.0))
    )
    for item in degraded_or_timeout:
        if len(selected) >= budget:
            break
        _append_if_new(item)

    if len(selected) < budget:
        by_score = sorted(
            results,
            key=lambda item: float(item.get("grade", {}).get("aggregate_score", 0.0)),
        )
        for item in by_score:
            if len(selected) >= budget:
                break
            _append_if_new(item)

    return selected[:budget]


def _rank_blocks_by_rca_impact(
    grouped: dict[str, list[BlockAnalysisSchema]],
) -> list[tuple[float, str]]:
    ranked: list[tuple[float, str]] = []
    for block_id, analyses in grouped.items():
        if not analyses:
            continue
        impact = sum(
            float(analysis.generic_issue_score) + float(analysis.criteria_misalignment_score)
            for analysis in analyses
        )
        ranked.append((impact, block_id))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def prompt_improvements(
    current_prompts: dict[str, str],
    block_analyses: list[BlockAnalysisSchema],
    block_details_map: dict[str, dict[str, Any]],
    mutation_block_budget: int = 3,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    grouped: dict[str, list[BlockAnalysisSchema]] = defaultdict(list)
    for analysis in block_analyses:
        if analysis.need_fix:
            grouped[analysis.block_id].append(analysis)

    improvements: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], tuple[PromptCriteriaScoreSchema, str]] = {}
    block_budget = max(0, int(mutation_block_budget))
    ranked_blocks = _rank_blocks_by_rca_impact(grouped)
    target_block_ids = [block_id for _, block_id in ranked_blocks]
    if block_budget > 0:
        target_block_ids = target_block_ids[:block_budget]
    else:
        target_block_ids = []

    target_block_id_set = set(target_block_ids)
    skipped_by_budget = [block_id for _, block_id in ranked_blocks if block_id not in target_block_id_set]
    for block_id in skipped_by_budget:
        if block_id in current_prompts:
            improvements[block_id] = current_prompts[block_id]
        diagnostics.append(
            {
                "block_id": block_id,
                "analysis_count": len(grouped.get(block_id, [])),
                "mutation_model": None,
                "changed": False,
                "accepted": False,
                "decision_reason": "skipped_by_mutation_budget",
                "required_placeholders_count": len(_placeholder_tokens(current_prompts.get(block_id, ""))),
                "missing_placeholders": [],
                "baseline_total": None,
                "candidate_total": None,
                "delta_total": 0.0,
                "baseline_scores": {},
                "candidate_scores": {},
                "score_models": {},
                "candidate_prompt_preview": "",
            }
        )

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
            "You are an expert prompt engineer. Improve this block prompt using a structured edit.\n"
            "Preserve baseline section scaffolding and placeholders unless explicitly required to fix schema fidelity.\n"
            "Do not insert specific training-case facts, numbers, or direct answer fragments.\n\n"
            f"Block ID: {block_id}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"RCA analyses:\n{analyses_text}\n\n"
            f"Prompt creation parameters:\n{_safe_json_dumps(criteria, max_chars=2500)}\n\n"
            f"Required placeholders to preserve: {required_placeholders}\n"
            "Editing protocol:\n"
            "- Keep existing section headings and ordering where possible.\n"
            "- Modify only sections necessary to address RCA findings.\n"
            "- Do not remove output schema instructions or contract sections.\n"
            "- Keep wording concise and generalizable.\n"
            "- Preserve strict JSON/output contract language for schema-based blocks.\n"
            "Return only the revised prompt text in `prompt`."
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
            logger.warning(
                f"Prompt mutation for {block_id} violated hard prompt contract {contract_hard_issues}; keeping baseline"
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
                    "contract_hard_issues": contract_hard_issues,
                    "contract_soft_issues": contract_soft_issues,
                    "contract_auto_repair_applied": auto_repair_applied,
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

        try:
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
        except Exception as exc:
            logger.exception("Mutation scoring failed for block=%s: %s", block_id, exc)
            improvements[block_id] = current_prompt
            diagnostics.append(
                {
                    "block_id": block_id,
                    "analysis_count": len(unique_analyses),
                    "mutation_model": used_model,
                    "changed": True,
                    "accepted": False,
                    "decision_reason": "mutation_scoring_error",
                    "error": str(exc),
                    "contract_issues": contract_issues,
                    "contract_hard_issues": contract_hard_issues,
                    "contract_soft_issues": contract_soft_issues,
                    "contract_auto_repair_applied": auto_repair_applied,
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

        delta_total = candidate_total - baseline_total
        acceptance_floor = baseline_total / PROMPT_SCORE_ACCEPTANCE_RATIO
        accepted = candidate_total >= acceptance_floor
        if accepted:
            improvements[block_id] = candidate_prompt
            if candidate_total >= baseline_total:
                decision_reason = "candidate_score_ge_baseline"
            else:
                decision_reason = "candidate_score_within_tolerance"
        else:
            improvements[block_id] = current_prompt
            decision_reason = "candidate_score_lt_tolerance"

        diagnostics.append(
            {
                "block_id": block_id,
                "analysis_count": len(unique_analyses),
                "mutation_model": used_model,
                "changed": True,
                "accepted": accepted,
                "decision_reason": decision_reason,
                "contract_issues": contract_issues,
                "contract_hard_issues": contract_hard_issues,
                "contract_soft_issues": contract_soft_issues,
                "contract_auto_repair_applied": auto_repair_applied,
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

    accepted_any = any(bool(item.get("accepted")) for item in diagnostics if item.get("changed"))
    if not accepted_any:
        fallback_grouped = {block_id: grouped.get(block_id, []) for block_id in target_block_ids}
        fallback_block_id = _select_fallback_block_id(fallback_grouped, current_prompts)
        if fallback_block_id:
            base_prompt = current_prompts.get(fallback_block_id, "")
            fallback_addendum = _deterministic_fallback_addendum(fallback_block_id).strip()
            normalized_base = _normalize_contract_text(base_prompt)
            normalized_addendum = _normalize_contract_text(fallback_addendum)

            if fallback_addendum and normalized_addendum not in normalized_base:
                fallback_prompt = base_prompt.rstrip() + "\n\n" + fallback_addendum
                fallback_details = block_details_map.get(fallback_block_id, {})
                fallback_issues = _validate_prompt_contract(
                    fallback_block_id,
                    fallback_prompt,
                    fallback_details,
                )
                fallback_hard, fallback_soft = _split_contract_issues(fallback_issues)
                fallback_repair_applied: list[str] = []
                if fallback_soft and not fallback_hard:
                    repaired_prompt, fallback_repair_applied = _auto_repair_contract_issues(
                        fallback_block_id,
                        fallback_prompt,
                        fallback_soft,
                    )
                    if repaired_prompt != fallback_prompt:
                        fallback_prompt = repaired_prompt
                        fallback_issues = _validate_prompt_contract(
                            fallback_block_id,
                            fallback_prompt,
                            fallback_details,
                        )
                        fallback_hard, fallback_soft = _split_contract_issues(fallback_issues)

                if not fallback_hard:
                    try:
                        baseline_score, baseline_score_model = _score_cached(
                            fallback_block_id,
                            base_prompt,
                            fallback_details,
                        )
                        candidate_score, candidate_score_model = _score_cached(
                            fallback_block_id,
                            fallback_prompt,
                            fallback_details,
                        )

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
                        acceptance_floor = baseline_total / PROMPT_SCORE_ACCEPTANCE_RATIO
                        fallback_accepted = candidate_total >= acceptance_floor
                        if fallback_accepted:
                            improvements[fallback_block_id] = fallback_prompt

                        diagnostics.append(
                            {
                                "block_id": fallback_block_id,
                                "analysis_count": len(fallback_grouped.get(fallback_block_id, [])),
                                "mutation_model": "deterministic_fallback",
                                "changed": True,
                                "accepted": fallback_accepted,
                                "decision_reason": (
                                    "fallback_candidate_score_ge_baseline"
                                    if fallback_accepted and candidate_total >= baseline_total
                                    else "fallback_candidate_score_within_tolerance"
                                    if fallback_accepted
                                    else "fallback_candidate_score_lt_tolerance"
                                ),
                                "contract_issues": fallback_issues,
                                "contract_hard_issues": fallback_hard,
                                "contract_soft_issues": fallback_soft,
                                "contract_auto_repair_applied": fallback_repair_applied,
                                "required_placeholders_count": len(_placeholder_tokens(base_prompt)),
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
                                "candidate_prompt_preview": fallback_prompt[:240],
                            }
                        )
                    except Exception as exc:
                        logger.exception(
                            "Fallback mutation scoring failed for block=%s: %s",
                            fallback_block_id,
                            exc,
                        )
                        diagnostics.append(
                            {
                                "block_id": fallback_block_id,
                                "analysis_count": len(fallback_grouped.get(fallback_block_id, [])),
                                "mutation_model": "deterministic_fallback",
                                "changed": True,
                                "accepted": False,
                                "decision_reason": "mutation_scoring_error",
                                "error": str(exc),
                                "contract_issues": fallback_issues,
                                "contract_hard_issues": fallback_hard,
                                "contract_soft_issues": fallback_soft,
                                "contract_auto_repair_applied": fallback_repair_applied,
                                "required_placeholders_count": len(_placeholder_tokens(base_prompt)),
                                "missing_placeholders": [],
                                "baseline_total": None,
                                "candidate_total": None,
                                "delta_total": 0.0,
                                "baseline_scores": {},
                                "candidate_scores": {},
                                "score_models": {},
                                "candidate_prompt_preview": fallback_prompt[:240],
                            }
                        )
                else:
                        diagnostics.append(
                            {
                                "block_id": fallback_block_id,
                                "analysis_count": len(fallback_grouped.get(fallback_block_id, [])),
                                "mutation_model": "deterministic_fallback",
                                "changed": True,
                                "accepted": False,
                            "decision_reason": "fallback_prompt_contract_violation",
                            "contract_issues": fallback_issues,
                            "contract_hard_issues": fallback_hard,
                            "contract_soft_issues": fallback_soft,
                            "contract_auto_repair_applied": fallback_repair_applied,
                            "required_placeholders_count": len(_placeholder_tokens(base_prompt)),
                            "missing_placeholders": [],
                            "baseline_total": None,
                            "candidate_total": None,
                            "delta_total": 0.0,
                            "baseline_scores": {},
                            "candidate_scores": {},
                            "score_models": {},
                            "candidate_prompt_preview": fallback_prompt[:240],
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
    reference_scores: list[float] | None = None,
    early_stop_min_cases: int = EARLY_STOP_MIN_CASES,
    early_stop_mean_delta_threshold: float = EARLY_STOP_MEAN_DELTA_THRESHOLD,
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

    total_cases = len(selected_cases)
    with prompt_override_context(prompt_suite):
        pipeline = Pipeline(
            enable_image_embedding=False,
            allow_visual_outputs=False,
        )

        for index, case in enumerate(selected_cases, start=1):
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
                "total": total_cases,
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
                    message=f"Running case {index}/{total_cases}",
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
                        message=f"Case {index}/{total_cases} timed out",
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
                        message=f"Case {index}/{total_cases} failed",
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

    return results, scores, {
        "case_count_requested": total_cases,
        "case_count_evaluated": len(scores),
        "timeout_enforcement_mode": timeout_enforcement_mode,
        "early_stop": early_stop,
    }


def train_ab_loop(
    base_prompts_path: str,
    output_path: str,
    test_cases_path: str,
    epochs: int = 10,
    num_test_cases_per_trial: int = 6,
    random_seed: int = 42,
    thinking_level: Literal["low", "med-synth", "med-plan", "high"] = "med-synth",
    fail_threshold: float = 7.5,
    holdout_sample_size: int = 6,
    rca_case_budget: int = 3,
    mutation_block_budget: int = 3,
    generalizer_cadence: int = 3,
    bootstrap_resamples: int = 1000,
    holdout_split_ratio: float = DEFAULT_HOLDOUT_SPLIT_RATIO,
    progress_status_path: str = "data/training_status.json",
    progress_events_path: str = "data/training_events.jsonl",
    track_progress: bool = True,
) -> dict[str, Any]:
    random.seed(random_seed)
    rng = random.Random(random_seed)

    current_prompts = load_prompts_file(base_prompts_path)
    test_cases = load_test_cases(test_cases_path)
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

    store = PromptSuiteStore(output_path)
    run_id = str(uuid.uuid4())
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
    store.save_generation(current_prompts, generation, {"note": "initial", "run_id": run_id, **split_metadata})
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
            "holdout_sample_size": holdout_sample_size,
            "thinking_level": thinking_level,
            "fail_threshold": fail_threshold,
            "random_seed": random_seed,
            "total_available_cases": len(test_cases),
            "sample_strategy": "stratified_rotating_unseen_first",
            "rca_case_budget": int(max(0, rca_case_budget)),
            "mutation_block_budget": int(max(0, mutation_block_budget)),
            "generalizer_cadence": int(max(0, generalizer_cadence)),
            "bootstrap_resamples": int(max(1, bootstrap_resamples)),
            "train_split_size": len(train_pool),
            "holdout_split_size": len(holdout_pool),
            "feature_flags": {
                "ENABLE_CANDIDATE_GATES": ENABLE_CANDIDATE_GATES,
                "ENABLE_TOOL_PROMPT_TEMPLATES": ENABLE_TOOL_PROMPT_TEMPLATES,
                "TRAINING_CASE_TIME_BUDGET_SECONDS": TRAINING_CASE_TIME_BUDGET_SECONDS,
                "STALL_WARNING_THRESHOLD_SECONDS": STALL_WARNING_THRESHOLD_SECONDS,
            },
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

            selected = sample_cases_stratified(
                sample_state=train_sampler_state,
                sample_size=min(num_test_cases_per_trial, len(train_pool)),
                rng=rng,
            )
            tracker.emit(
                event_type="sample_selected",
                phase="sampling",
                step="select_cases",
                message=f"Selected {len(selected)} case(s) for epoch {epoch_current}",
                epoch_current=epoch_current,
                payload={
                    "sampled_case_ids": [case.get("id") for case in selected],
                    "sampled_categories": sorted({case.get("category") for case in selected if case.get("category")}),
                    "sample_strategy": "stratified_rotating_unseen_first",
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
            results_a, scores_a, phase_a_meta = evaluate_suite(
                current_prompts,
                selected,
                thinking_level,
                phase="phase_a_eval",
                epoch_current=epoch_current,
                progress_tracker=tracker,
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

            failed = _select_rca_cases(results_a, rca_case_budget=rca_case_budget)
            all_analyses: list[BlockAnalysisSchema] = []
            valid_block_ids = sorted(list(current_prompts.keys()))
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
                    "rca_case_ids": rca_case_ids,
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
                    )
                all_analyses.extend(analyses)
                tracker.emit(
                    event_type="rca_completed",
                    phase="rca",
                    step="case_rca_done",
                    message=f"Epoch {epoch_current}: RCA {failed_index}/{len(failed)} complete",
                    epoch_current=epoch_current,
                    payload={
                        "analyses_added": len(analyses),
                        "implicated_blocks": implicated_block_ids,
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
                },
            )
            if tracker is not None:
                with tracker.active_call(
                    "prompt_improvements",
                    metadata={"epoch": epoch_current, "rca_items": len(all_analyses)},
                ):
                    improvements, prompt_scoring = prompt_improvements(
                        current_prompts=current_prompts,
                        block_analyses=all_analyses,
                        block_details_map=block_details_map,
                        mutation_block_budget=mutation_block_budget,
                    )
            else:
                improvements, prompt_scoring = prompt_improvements(
                    current_prompts=current_prompts,
                    block_analyses=all_analyses,
                    block_details_map=block_details_map,
                    mutation_block_budget=mutation_block_budget,
                )
            candidate_prompts.update(improvements)

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
                in {
                    "candidate_score_lt_baseline",
                    "candidate_score_lt_tolerance",
                    "fallback_candidate_score_lt_tolerance",
                }
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
                },
                metrics={
                    "prompt_scored_blocks": len(scored_blocks),
                    "prompt_accepted_blocks": len(accepted_blocks),
                    "prompt_rejected_blocks": len(rejected_blocks),
                    "prompt_changed_blocks": len(changed_blocks),
                    "prompt_contract_rejected_blocks": len(contract_rejected_blocks),
                    "prompt_score_rejected_blocks": len(score_rejected_blocks),
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

            generalizer = None
            if int(max(0, generalizer_cadence)) > 0 and (epoch_current % int(max(1, generalizer_cadence)) == 0):
                tracker.emit(
                    event_type="generalizer_started",
                    phase="generalizer_check",
                    step="run_generalizer",
                    message=f"Epoch {epoch_current}: running generalizer check",
                    epoch_current=epoch_current,
                )
                if tracker is not None:
                    with tracker.active_call(
                        "generalizer_check",
                        metadata={"epoch": epoch_current},
                    ):
                        generalizer = generalizer_check(candidate_prompts, selected)
                else:
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
                )
                avg_score_b = sum(scores_b) / max(len(scores_b), 1)
                stats_b = _score_stats(scores_b)
                phase_b_eval_summary = _summarize_eval_results(results_b)
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
                }
                tracker.emit(
                    event_type="phase_skipped",
                    phase="phase_b_eval",
                    step="evaluate_candidate_skipped",
                    message=f"Epoch {epoch_current}: candidate evaluation skipped (no changes)",
                    epoch_current=epoch_current,
                )

            winner = "baseline"
            train_delta_stats = _paired_delta_stats(
                scores_a,
                scores_b,
                bootstrap_resamples=bootstrap_resamples,
                rng=rng,
            )
            improvement_delta = float(train_delta_stats.get("mean_delta", avg_score_b - avg_score_a))
            prompt_delta_avg = (sum(delta_totals) / len(delta_totals)) if delta_totals else 0.0

            gate_results = _evaluate_candidate_gates(
                delta_stats=train_delta_stats,
                phase_a_summary=phase_a_eval_summary,
                phase_b_summary=phase_b_eval_summary,
                enabled=ENABLE_CANDIDATE_GATES,
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
                        sample_size=min(holdout_sample_size, len(holdout_pool)),
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
                            "sample_strategy": "stratified_rotating_unseen_first",
                        },
                    )
                    holdout_results_a, holdout_scores_a, holdout_meta_a = evaluate_suite(
                        current_prompts,
                        selected_holdout,
                        thinking_level,
                        phase="holdout_phase_a_eval",
                        epoch_current=epoch_current,
                        progress_tracker=tracker,
                    )
                    holdout_results_b, holdout_scores_b, holdout_meta_b = evaluate_suite(
                        candidate_prompts,
                        selected_holdout,
                        thinking_level,
                        phase="holdout_phase_b_eval",
                        epoch_current=epoch_current,
                        progress_tracker=tracker,
                    )
                    holdout_phase_a_summary = _summarize_eval_results(holdout_results_a)
                    holdout_phase_b_summary = _summarize_eval_results(holdout_results_b)
                    holdout_delta_stats = _paired_delta_stats(
                        holdout_scores_a,
                        holdout_scores_b,
                        bootstrap_resamples=bootstrap_resamples,
                        rng=rng,
                    )
                    holdout_confirmation = _evaluate_holdout_confirmation(holdout_delta_stats)
                    if not holdout_confirmation.get("passed"):
                        gate_failure_reasons.append("holdout_confirmation_gate")
                else:
                    holdout_confirmation = {
                        "passed": True,
                        "mean_delta": 0.0,
                        "ci_lower": 0.0,
                        "ci_upper": 0.0,
                        "rule": "holdout split unavailable",
                    }

                if holdout_confirmation and holdout_confirmation.get("passed"):
                    winner = "candidate"

            if winner == "candidate":
                generation += 1
                current_prompts = candidate_prompts

            selection_stats = {
                "train_delta_stats": train_delta_stats,
                "holdout_delta_stats": holdout_delta_stats,
                "holdout_confirmation": holdout_confirmation,
                "early_stop_triggered": early_stop_triggered,
                "prompt_delta_avg_total": prompt_delta_avg,
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
                    "phase_a_eval_summary": phase_a_eval_summary,
                    "phase_b_eval_summary": phase_b_eval_summary,
                    "selection_stats": selection_stats,
                },
                metrics={
                    "candidate_gates_enabled": 1 if ENABLE_CANDIDATE_GATES else 0,
                    "candidate_gate_passed": 1 if winner == "candidate" else 0,
                    "candidate_runtime_mean_ratio": gate_results.get("ratios", {}).get("mean_case_time_ratio") or 0.0,
                    "candidate_runtime_p90_ratio": gate_results.get("ratios", {}).get("p90_case_time_ratio") or 0.0,
                    "candidate_prompt_delta_avg_total": prompt_delta_avg,
                },
            )

            metadata = {
                "epoch": epoch,
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
                "scores_a": scores_a,
                "scores_b": scores_b,
                "thinking_level": thinking_level,
                "num_failures_in_a": len(failed),
                "num_rca_items": len(all_analyses),
                "generalizer": generalizer,
                "phase_a_eval_summary": phase_a_eval_summary,
                "phase_b_eval_summary": phase_b_eval_summary,
                "phase_a_eval_meta": phase_a_meta,
                "phase_b_eval_meta": phase_b_meta,
                "holdout_phase_a_eval_summary": holdout_phase_a_summary,
                "holdout_phase_b_eval_summary": holdout_phase_b_summary,
                "holdout_phase_a_eval_meta": holdout_meta_a,
                "holdout_phase_b_eval_meta": holdout_meta_b,
                "candidate_gate_results": gate_results,
                "candidate_gate_failure_reasons": gate_failure_reasons,
                "prompt_scoring": prompt_scoring,
                "mutation_rejection_breakdown": rejection_breakdown,
                "mutation_rejection_issue_matrix": rejection_issue_matrix,
                "sample_strategy": "stratified_rotating_unseen_first",
                "train_case_ids": [_case_identifier(case) for case in selected],
                "holdout_case_ids": holdout_case_ids,
                "rca_case_ids": rca_case_ids,
                "selection_stats": selection_stats,
                "train_split": split_metadata["train_split"],
                "holdout_split": split_metadata["holdout_split"],
                "early_stop": phase_b_meta.get("early_stop", {"triggered": False}),
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
            "sample_strategy": "stratified_rotating_unseen_first",
            "train_split_size": len(train_pool),
            "holdout_split_size": len(holdout_pool),
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
    num_test_cases_per_trial: int = 6,
    holdout_sample_size: int = 6,
    rca_case_budget: int = 3,
    mutation_block_budget: int = 3,
    generalizer_cadence: int = 3,
    bootstrap_resamples: int = 1000,
    holdout_split_ratio: float = DEFAULT_HOLDOUT_SPLIT_RATIO,
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
        holdout_sample_size=holdout_sample_size,
        rca_case_budget=rca_case_budget,
        mutation_block_budget=mutation_block_budget,
        generalizer_cadence=generalizer_cadence,
        bootstrap_resamples=bootstrap_resamples,
        holdout_split_ratio=holdout_split_ratio,
        random_seed=random_seed,
        thinking_level=thinking_level,
        progress_status_path=progress_status_path,
        progress_events_path=progress_events_path,
        track_progress=track_progress,
    )


if __name__ == "__main__":
    result = run_training_loop()
    print(json.dumps(result, indent=2))
