from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import prompt_training as pt


def _make_case(case_id: str, category: str, difficulty: str) -> dict[str, str]:
    return {
        "id": case_id,
        "category": category,
        "difficulty": difficulty,
        "prompt": f"Prompt for {case_id}",
        "validation": "Validation text",
    }


def test_stratified_split_preserves_bucket_coverage() -> None:
    cases: list[dict[str, str]] = []
    for idx in range(5):
        cases.append(_make_case(f"math-{idx}", "math", "hard"))
    for idx in range(5):
        cases.append(_make_case(f"sci-{idx}", "science", "very_hard"))

    train, holdout = pt.stratified_split_cases(cases, holdout_ratio=0.2, rng=random.Random(13))
    assert len(train) + len(holdout) == len(cases)
    assert len(train) > 0
    assert len(holdout) > 0

    train_buckets = pt.build_case_buckets(train)
    holdout_buckets = pt.build_case_buckets(holdout)
    assert "math::hard" in train_buckets and "math::hard" in holdout_buckets
    assert "science::very_hard" in train_buckets and "science::very_hard" in holdout_buckets


def test_stratified_sampling_rotates_unseen_first() -> None:
    cases = [
        _make_case("a-1", "a", "hard"),
        _make_case("a-2", "a", "hard"),
        _make_case("a-3", "a", "hard"),
        _make_case("b-1", "b", "hard"),
        _make_case("b-2", "b", "hard"),
        _make_case("b-3", "b", "hard"),
    ]
    rng = random.Random(99)
    state = pt._build_sampling_state(cases, rng)

    first = pt.sample_cases_stratified(sample_state=state, sample_size=3, rng=rng)
    second = pt.sample_cases_stratified(sample_state=state, sample_size=3, rng=rng)

    first_ids = {_case["id"] for _case in first}
    second_ids = {_case["id"] for _case in second}
    assert len(first_ids) == 3
    assert len(second_ids) == 3
    assert first_ids.isdisjoint(second_ids)


def test_select_rca_cases_prioritizes_degraded_and_budget() -> None:
    results = [
        {
            "case": {"id": "high-degraded", "prompt": "p"},
            "grade": {"aggregate_score": 8.9},
            "case_stats": {"degraded_mode_active": True, "timed_out": False},
        },
        {
            "case": {"id": "timeout", "prompt": "p"},
            "grade": {"aggregate_score": 9.2},
            "case_stats": {"degraded_mode_active": False, "timed_out": True},
        },
        {
            "case": {"id": "low-score-1", "prompt": "p"},
            "grade": {"aggregate_score": 2.0},
            "case_stats": {"degraded_mode_active": False, "timed_out": False},
        },
        {
            "case": {"id": "low-score-2", "prompt": "p"},
            "grade": {"aggregate_score": 3.0},
            "case_stats": {"degraded_mode_active": False, "timed_out": False},
        },
    ]
    selected = pt._select_rca_cases(results, rca_case_budget=3)
    selected_ids = [item["case"]["id"] for item in selected]
    assert len(selected_ids) == 3
    assert "high-degraded" in selected_ids
    assert "timeout" in selected_ids
    assert "low-score-1" in selected_ids


def test_candidate_gate_uses_effect_size_and_uncertainty() -> None:
    baseline = [6.0, 6.0, 6.0, 6.0, 6.0]
    candidate = [6.4, 6.5, 6.6, 6.5, 6.4]
    delta_stats = pt._paired_delta_stats(
        baseline,
        candidate,
        bootstrap_resamples=200,
        rng=random.Random(3),
    )
    gate = pt._evaluate_candidate_gates(
        delta_stats=delta_stats,
        phase_a_summary={"mean_case_time_s": 100.0, "p90_case_time_s": 150.0, "avg_tool_failure_signals": 0.2, "degraded_case_rate": 0.1},
        phase_b_summary={"mean_case_time_s": 110.0, "p90_case_time_s": 170.0, "avg_tool_failure_signals": 0.3, "degraded_case_rate": 0.12},
        enabled=True,
    )
    assert gate["all_passed"] is True


def test_early_stop_candidate_eval_triggered() -> None:
    triggered, info = pt._should_early_stop_candidate_eval(
        baseline_scores=[7.0, 7.0, 7.0, 7.0],
        candidate_scores=[6.0, 6.0, 6.0, 6.0],
        min_cases=4,
        mean_delta_threshold=-0.35,
    )
    assert triggered is True
    assert info["running_mean_delta"] < -0.35


def test_postprocess_grade_adds_deterministic_and_penalty_clamp() -> None:
    case = {
        "id": "math-ans",
        "prompt": "Find optimum",
        "validation": "Must show answer",
        "answer": "64",
    }
    run_output = {
        "response": "The maximum is 64.",
        "degraded_mode_active": True,
        "tool_failure_signals": [{"error": "x"}, {"error": "y"}],
        "tool_context": {},
    }
    grade = {
        "aggregate_score": 1.2,
        "prompt_alignment_score": 2,
        "factuality_score": 2,
        "clarity_score": 2,
        "helpfulness_score": 2,
        "safety_score": 2,
        "tool_usage_score": 2,
        "format_quality_score": 2,
        "engagement_score": 2,
        "citation_quality_score": 2,
    }
    processed = pt._postprocess_grade(case=case, run_output=run_output, grade=grade, timed_out=False)
    assert processed["deterministic_answer_score"] is not None
    assert processed["aggregate_score"] >= 1.0
    assert processed["reliability_penalty"] > 0.0


def test_prompt_improvements_handles_mutation_scoring_error(monkeypatch: pytest.MonkeyPatch) -> None:
    current_prompts = {"custom_block": "Original prompt with {{{objective}}}"}
    block_analyses = [
        pt.BlockAnalysisSchema(
            block_id="custom_block",
            generic_issue_score=8,
            criteria_misalignment_score=8,
            need_fix=True,
            analysis="Issue",
            what_to_fix="Fix this",
        )
    ]
    block_details_map = {
        "custom_block": {
            "schema": str,
            "prompt_creation_parameters": {},
        }
    }

    def _mock_mutation(*args, **kwargs):  # type: ignore[no-untyped-def]
        return pt.PromptMutationSchema(plan_for_improvement="p", prompt="Mutated {{{objective}}}"), "mock"

    def _mock_score(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(pt, "_generate_with_oss_fallback", _mock_mutation)
    monkeypatch.setattr(pt, "score_prompt_against_criteria", _mock_score)

    improvements, diagnostics = pt.prompt_improvements(
        current_prompts=current_prompts,
        block_analyses=block_analyses,
        block_details_map=block_details_map,
        mutation_block_budget=1,
    )
    assert improvements["custom_block"] == current_prompts["custom_block"]
    assert any(item.get("decision_reason") == "mutation_scoring_error" for item in diagnostics)


def test_timeout_enforcement_mode_reports_expected_values() -> None:
    assert pt._timeout_enforcement_mode(0) == "disabled_no_budget"
    assert pt._timeout_enforcement_mode(10) in {
        "signal_alarm",
        "disabled_no_sigalrm",
        "disabled_non_main_thread",
    }
