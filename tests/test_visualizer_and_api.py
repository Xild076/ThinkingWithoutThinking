from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from prompt_improvement_visualizer import create_prompt_improvement_visualization


def _write_generation_fixture(path: Path) -> None:
    payload = {
        "generations": [
            {
                "generation": 0,
                "timestamp": 10.0,
                "prompts": {},
                "metadata": {
                    "run_id": "run-old",
                    "epoch": 0,
                    "avg_score_a": 6.0,
                    "avg_score_b": 6.1,
                    "improvement_delta": 0.1,
                    "winner": "candidate",
                    "scores_a": [6.0, 6.0],
                    "scores_b": [6.1, 6.2],
                    "train_case_ids": ["c1", "c2"],
                    "candidate_gate_results": {
                        "gates": {
                            "quality_gate": {"passed": True},
                            "runtime_gate": {"passed": True},
                            "stability_gate": {"passed": True},
                        }
                    },
                    "phase_a_eval_summary": {"degraded_case_rate": 0.0},
                    "phase_b_eval_summary": {"degraded_case_rate": 0.0},
                    "prompt_scoring_summary": {
                        "changed_blocks": 1,
                        "accepted_blocks": 1,
                        "rejected_blocks": 0,
                    },
                    "selection_stats": {
                        "train_delta_stats": {
                            "mean_delta": 0.15,
                            "ci_lower": 0.05,
                            "ci_upper": 0.25,
                            "pair_count": 2,
                        }
                    },
                },
            },
            {
                "generation": 0,
                "timestamp": 20.0,
                "prompts": {},
                "metadata": {
                    "run_id": "run-new",
                    "epoch": 0,
                    "avg_score_a": 5.8,
                    "avg_score_b": 5.7,
                    "improvement_delta": -0.1,
                    "winner": "baseline",
                    "scores_a": [5.8, 5.8],
                    "scores_b": [5.6, 5.8],
                    "train_case_ids": ["c3", "c4"],
                    "candidate_gate_failure_reasons": ["quality_gate"],
                    "candidate_gate_results": {
                        "gates": {
                            "quality_gate": {"passed": False},
                            "runtime_gate": {"passed": True},
                            "stability_gate": {"passed": True},
                        }
                    },
                    "phase_a_eval_summary": {"degraded_case_rate": 0.0},
                    "phase_b_eval_summary": {"degraded_case_rate": 0.5},
                    "prompt_scoring_summary": {
                        "changed_blocks": 2,
                        "accepted_blocks": 0,
                        "rejected_blocks": 2,
                    },
                    "selection_stats": {
                        "train_delta_stats": {
                            "mean_delta": -0.1,
                            "ci_lower": -0.2,
                            "ci_upper": 0.0,
                            "pair_count": 2,
                        }
                    },
                },
            },
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_visualizer_latest_run_and_index(tmp_path: Path) -> None:
    input_path = tmp_path / "generations.json"
    out_dir = tmp_path / "plots"
    _write_generation_fixture(input_path)

    batch = create_prompt_improvement_visualization(
        input_path=str(input_path),
        output_dir=str(out_dir),
        latest_run=True,
        emit_index=True,
        golden_alias=False,
    )

    payload = batch.to_dict()
    assert payload["runs_count"] == 1
    assert payload["runs"][0]["run_id"] == "run-new"
    assert payload["golden_png_path"] == ""
    assert Path(payload["index_json_path"]).exists()
    assert Path(payload["index_html_path"]).exists()

    html_path = Path(payload["runs"][0]["html_path"])
    assert "One-epoch run (not truncation)." in html_path.read_text(encoding="utf-8")


def test_training_run_history_endpoints(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import app as app_module

    input_path = tmp_path / "generations.json"
    _write_generation_fixture(input_path)

    class _PipelineStub:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(app_module, "TRAINING_GENERATIONS_PATH", input_path)
    monkeypatch.setattr(app_module, "Pipeline", _PipelineStub)

    with TestClient(app_module.app) as client:
        runs_payload = client.get("/training/runs").json()
        assert runs_payload["status"] == "ok"
        assert runs_payload["runs_count"] == 2

        run_id = runs_payload["runs"][0]["run_id"]
        summary_payload = client.get(f"/training/runs/{run_id}/summary").json()
        assert summary_payload["status"] == "ok"

        epochs_payload = client.get(f"/training/runs/{run_id}/epochs").json()
        assert epochs_payload["status"] == "ok"
        assert epochs_payload["epochs_count"] == 1

        cases_payload = client.get(f"/training/runs/{run_id}/epoch/0/cases").json()
        assert cases_payload["status"] == "ok"
        assert cases_payload["pair_count"] == 2
