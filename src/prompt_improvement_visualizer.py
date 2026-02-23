from __future__ import annotations

import base64
import json
import math
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    pass


ENABLE_VISUALIZER_V2 = str(os.getenv("ENABLE_VISUALIZER_V2", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass
class VisualizationResult:
    run_id: str
    history_path: str
    output_dir: str
    png_path: str
    html_path: str
    summary_path: str
    epochs_count: int
    improved_epochs: int
    accepted_prompt_updates: int
    net_improvement_delta: float
    quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "run_id": self.run_id,
            "history_path": self.history_path,
            "output_dir": self.output_dir,
            "png_path": self.png_path,
            "html_path": self.html_path,
            "summary_path": self.summary_path,
            "epochs_count": self.epochs_count,
            "improved_epochs": self.improved_epochs,
            "accepted_prompt_updates": self.accepted_prompt_updates,
            "net_improvement_delta": self.net_improvement_delta,
            "quality_score": self.quality_score,
        }


@dataclass
class VisualizationBatchResult:
    history_path: str
    output_dir: str
    generated_at: str
    runs: list[VisualizationResult]
    golden_run_id: str
    golden_png_path: str
    golden_html_path: str
    index_json_path: str | None = None
    index_html_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "history_path": self.history_path,
            "output_dir": self.output_dir,
            "generated_at": self.generated_at,
            "runs_count": len(self.runs),
            "runs": [run.to_dict() for run in self.runs],
            "golden_run_id": self.golden_run_id,
            "golden_png_path": self.golden_png_path,
            "golden_html_path": self.golden_html_path,
            "index_json_path": self.index_json_path,
            "index_html_path": self.index_html_path,
        }



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except Exception:
        return default



def _load_generations(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    generations = payload.get("generations", [])
    if not isinstance(generations, list):
        return []
    return [record for record in generations if isinstance(record, dict)]



def _records_by_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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
            key=lambda record: (
                int((record.get("metadata") or {}).get("epoch", -1)),
                float(record.get("timestamp") or 0.0),
            )
        )
    return grouped



def _epoch_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if "epoch" in metadata:
            entries.append(record)
    return entries



def _collect_block_ids(epoch_records: list[dict[str, Any]]) -> list[str]:
    block_ids: set[str] = set()
    for record in epoch_records:
        metadata = record.get("metadata") or {}
        scoring = metadata.get("prompt_scoring") or []
        if isinstance(scoring, list):
            for item in scoring:
                if not isinstance(item, dict):
                    continue
                block_id = item.get("block_id")
                if block_id:
                    block_ids.add(str(block_id))
    return sorted(block_ids)



def _build_rows(selected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _epoch_records(selected_records):
        metadata = record.get("metadata") or {}
        epoch = int(metadata.get("epoch", -1))
        scores_a = [float(value) for value in (metadata.get("scores_a") or []) if isinstance(value, (int, float))]
        scores_b = [float(value) for value in (metadata.get("scores_b") or []) if isinstance(value, (int, float))]
        pair_count = min(len(scores_a), len(scores_b))
        per_case_deltas = [scores_b[idx] - scores_a[idx] for idx in range(pair_count)]

        gate_results = metadata.get("candidate_gate_results") or {}
        gates = gate_results.get("gates") or {}
        prompt_scoring_summary = metadata.get("prompt_scoring_summary") or {}
        selection_stats = metadata.get("selection_stats") or {}
        train_delta_stats = selection_stats.get("train_delta_stats") or {}
        timeout_stats = metadata.get("timeout_stats") or {}
        phase_b_timeout = timeout_stats.get("phase_b") or {}
        generalizer = metadata.get("generalizer") or {}
        generalizer_candidate = generalizer.get("candidate") if isinstance(generalizer, dict) else {}

        rows.append(
            {
                "epoch": epoch,
                "winner": str(metadata.get("winner") or "unknown"),
                "avg_score_a": _safe_float(metadata.get("avg_score_a"), 0.0),
                "avg_score_b": _safe_float(metadata.get("avg_score_b"), 0.0),
                "improvement_delta": _safe_float(metadata.get("improvement_delta"), 0.0),
                "train_mean_delta": _safe_float(train_delta_stats.get("mean_delta"), 0.0),
                "train_ci_lower": _safe_float(train_delta_stats.get("ci_lower"), 0.0),
                "train_ci_upper": _safe_float(train_delta_stats.get("ci_upper"), 0.0),
                "pair_count": int(train_delta_stats.get("pair_count") or pair_count),
                "per_case_deltas": per_case_deltas,
                "phase_a_eval_summary": metadata.get("phase_a_eval_summary") or {},
                "phase_b_eval_summary": metadata.get("phase_b_eval_summary") or {},
                "paired_phase_a_eval_summary": metadata.get("paired_phase_a_eval_summary") or {},
                "paired_phase_b_eval_summary": metadata.get("paired_phase_b_eval_summary") or {},
                "candidate_gate_failure_reasons": metadata.get("candidate_gate_failure_reasons") or [],
                "quality_gate_passed": bool((gates.get("quality_gate") or {}).get("passed")),
                "runtime_gate_passed": bool((gates.get("runtime_gate") or {}).get("passed")),
                "stability_gate_passed": bool((gates.get("stability_gate") or {}).get("passed")),
                "changed_blocks": int(prompt_scoring_summary.get("changed_blocks") or 0),
                "accepted_blocks": int(prompt_scoring_summary.get("accepted_blocks") or 0),
                "rejected_blocks": int(prompt_scoring_summary.get("rejected_blocks") or 0),
                "mutation_rejection_breakdown": metadata.get("mutation_rejection_breakdown") or {},
                "mutation_retry_summary": metadata.get("mutation_retry_summary") or {},
                "mutation_attempts": metadata.get("mutation_attempts") or [],
                "rca_case_ids": metadata.get("rca_case_ids") or [],
                "mutated_block_ids": metadata.get("mutated_block_ids") or [],
                "events_archive_path": metadata.get("events_archive_path"),
                "early_stop": metadata.get("early_stop") or {},
                "timeout_count": int(phase_b_timeout.get("timeout_count") or 0),
                "timeout_rate": _safe_float(phase_b_timeout.get("timeout_rate"), 0.0),
                "generalizer_risk": _safe_float((generalizer_candidate or {}).get("overfit_risk_score"), 0.0),
                "generalizer_suspicious_delta": _safe_float(generalizer.get("suspicious_delta"), 0.0),
            }
        )

    rows.sort(key=lambda row: row["epoch"])
    return rows



def _build_block_delta_matrix(epoch_records: list[dict[str, Any]], block_ids: list[str]) -> tuple[list[list[float]], list[int]]:
    epoch_values: list[int] = []
    matrix: list[list[float]] = []
    block_index = {block_id: idx for idx, block_id in enumerate(block_ids)}

    for record in epoch_records:
        metadata = record.get("metadata") or {}
        epoch_values.append(int(metadata.get("epoch", -1)))
        row = [0.0 for _ in block_ids]
        scoring = metadata.get("prompt_scoring") or []
        if isinstance(scoring, list):
            for item in scoring:
                if not isinstance(item, dict):
                    continue
                block_id = str(item.get("block_id") or "")
                if block_id not in block_index:
                    continue
                baseline_total = _safe_float(item.get("baseline_total"), 0.0)
                candidate_total = _safe_float(item.get("candidate_total"), 0.0)
                row[block_index[block_id]] = candidate_total - baseline_total
        matrix.append(row)

    return matrix, epoch_values



def _run_quality_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("-inf")
    improved_epochs = sum(1 for row in rows if _safe_float(row.get("improvement_delta"), 0.0) > 0)
    accepted_epochs = sum(1 for row in rows if str(row.get("winner") or "") == "candidate")
    net_delta = sum(_safe_float(row.get("improvement_delta"), 0.0) for row in rows)
    avg_timeout_rate = sum(_safe_float(row.get("timeout_rate"), 0.0) for row in rows) / max(len(rows), 1)
    return (
        40.0 * improved_epochs
        + 35.0 * accepted_epochs
        + 25.0 * net_delta
        - 20.0 * avg_timeout_rate
        + 5.0 * math.log1p(len(rows))
    )



def _panel_gate_waterfall(ax, rows: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    quality = [1 if row.get("quality_gate_passed") else 0 for row in rows]
    runtime = [1 if row.get("runtime_gate_passed") else 0 for row in rows]
    stability = [1 if row.get("stability_gate_passed") else 0 for row in rows]

    ax.bar(epochs, quality, color="#2e7d32", alpha=0.7, label="quality")
    ax.bar(epochs, runtime, bottom=quality, color="#1565c0", alpha=0.7, label="runtime")
    stacked = [quality[idx] + runtime[idx] for idx in range(len(epochs))]
    ax.bar(epochs, stability, bottom=stacked, color="#ef6c00", alpha=0.75, label="stability")
    ax.set_ylim(0, 3.2)
    ax.set_title("Gate Waterfall per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3 gates pass"])
    ax.legend(loc="upper right", fontsize=8)



def _panel_mutation_funnel(ax, rows: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    attempted = [int(row.get("changed_blocks") or 0) for row in rows]
    accepted = [int(row.get("accepted_blocks") or 0) for row in rows]
    rejected = [int(row.get("rejected_blocks") or 0) for row in rows]
    retries = [
        int((row.get("mutation_retry_summary") or {}).get("total_retry_attempts") or 0)
        for row in rows
    ]

    ax.plot(epochs, attempted, marker="o", linewidth=2.0, label="attempted")
    ax.plot(epochs, accepted, marker="o", linewidth=2.0, label="accepted")
    ax.plot(epochs, rejected, marker="o", linewidth=2.0, label="rejected")
    ax.plot(epochs, retries, marker="x", linewidth=1.8, label="retry attempts")
    ax.set_title("Mutation Funnel")
    ax.set_xlabel("Epoch")
    ax.legend(loc="best", fontsize=8)



def _panel_rca_trace(ax, rows: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    rca_counts = [len(row.get("rca_case_ids") or []) for row in rows]
    mutated_counts = [len(row.get("mutated_block_ids") or []) for row in rows]
    ax.bar(epochs, rca_counts, alpha=0.6, label="RCA cases", color="#8e24aa")
    ax.plot(epochs, mutated_counts, marker="o", linewidth=2.0, label="mutated blocks", color="#00897b")
    ax.set_title("RCA -> Mutation Trace")
    ax.set_xlabel("Epoch")
    ax.legend(loc="best", fontsize=8)



def _panel_paired_deltas(ax, rows: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    means = [_safe_float(row.get("train_mean_delta"), 0.0) for row in rows]
    ci_low = [_safe_float(row.get("train_ci_lower"), 0.0) for row in rows]
    ci_high = [_safe_float(row.get("train_ci_upper"), 0.0) for row in rows]

    lower_err = [max(0.0, means[idx] - ci_low[idx]) for idx in range(len(epochs))]
    upper_err = [max(0.0, ci_high[idx] - means[idx]) for idx in range(len(epochs))]
    ax.errorbar(epochs, means, yerr=[lower_err, upper_err], fmt="o", capsize=5, color="#1e88e5", label="mean Δ with CI")

    for row in rows:
        epoch = int(row["epoch"])
        per_case = row.get("per_case_deltas") or []
        if not per_case:
            continue
        jitter_x = [epoch + ((idx % 3) - 1) * 0.05 for idx in range(len(per_case))]
        ax.scatter(jitter_x, per_case, s=18, alpha=0.35, color="#6d4c41")

    ax.axhline(0.0, color="#555", linewidth=1.0, alpha=0.7)
    ax.set_title("Paired Per-Case Delta Distribution + CI")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score Δ (B-A)")
    ax.legend(loc="best", fontsize=8)



def _panel_timeout_degraded(ax, rows: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    timeout_rate = [_safe_float(row.get("timeout_rate"), 0.0) for row in rows]
    degraded_rate = [
        _safe_float(((row.get("phase_b_eval_summary") or {}).get("degraded_case_rate")), 0.0)
        for row in rows
    ]
    early_stop = [1 if bool((row.get("early_stop") or {}).get("triggered")) else 0 for row in rows]

    ax.plot(epochs, timeout_rate, marker="o", linewidth=2.0, label="timeout rate")
    ax.plot(epochs, degraded_rate, marker="o", linewidth=2.0, label="degraded rate")
    ax.bar(epochs, early_stop, alpha=0.25, label="early stop", color="#c62828")
    ax.set_title("Timeout / Degraded Diagnostics")
    ax.set_xlabel("Epoch")
    ax.legend(loc="best", fontsize=8)



def _panel_block_heatmap(ax, block_ids: list[str], block_matrix: list[list[float]], matrix_epochs: list[int]) -> None:
    if block_ids and block_matrix:
        image = ax.imshow(block_matrix, aspect="auto", cmap="RdYlGn", interpolation="nearest", vmin=-3.0, vmax=3.0)
        ax.set_yticks(range(len(matrix_epochs)))
        ax.set_yticklabels([f"E{epoch}" for epoch in matrix_epochs])
        ax.set_xticks(range(len(block_ids)))
        ax.set_xticklabels(block_ids, rotation=35, ha="right", fontsize=8)
        ax.set_title("Per-Block Score Delta (candidate-baseline)")
        plt.colorbar(image, ax=ax)
    else:
        ax.text(0.5, 0.5, "No prompt_scoring block data available", ha="center", va="center")
        ax.set_title("Per-Block Score Delta")
        ax.set_xticks([])
        ax.set_yticks([])



def _plot_dashboard(
    rows: list[dict[str, Any]],
    block_ids: list[str],
    block_matrix: list[list[float]],
    matrix_epochs: list[int],
    run_id: str,
    quality_score: float,
):
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    ax_scores = axes[0][0]
    ax_gates = axes[0][1]
    ax_funnel = axes[1][0]
    ax_heat = axes[1][1]
    ax_deltas = axes[2][0]
    ax_timeout = axes[2][1]

    epochs = [int(row["epoch"]) for row in rows]
    avg_a = [row["avg_score_a"] for row in rows]
    avg_b = [row["avg_score_b"] for row in rows]
    deltas = [row["improvement_delta"] for row in rows]
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in deltas]

    ax_scores.plot(epochs, avg_a, marker="o", linewidth=2.2, label="Baseline (A)")
    ax_scores.plot(epochs, avg_b, marker="o", linewidth=2.4, label="Candidate (B)")
    ax_delta = ax_scores.twinx()
    ax_delta.bar(epochs, deltas, alpha=0.22, color=colors, label="Improvement Δ (B-A)")
    ax_delta.axhline(0.0, color="#666", linewidth=1.0, alpha=0.7)
    ax_scores.set_title("Score Trend + Improvement Delta")
    ax_scores.set_xlabel("Epoch")
    ax_scores.set_ylabel("Average score")
    ax_delta.set_ylabel("Delta")
    lines_scores, labels_scores = ax_scores.get_legend_handles_labels()
    lines_delta, labels_delta = ax_delta.get_legend_handles_labels()
    ax_scores.legend(lines_scores + lines_delta, labels_scores + labels_delta, loc="best", fontsize=8)

    _panel_gate_waterfall(ax_gates, rows)
    _panel_mutation_funnel(ax_funnel, rows)
    _panel_block_heatmap(ax_heat, block_ids, block_matrix, matrix_epochs)
    _panel_paired_deltas(ax_deltas, rows)
    _panel_timeout_degraded(ax_timeout, rows)

    final_delta = deltas[-1] if deltas else 0.0
    fig.suptitle(
        f"Prompt Improvement Dashboard (V2) - run={run_id} - quality={quality_score:.2f} - final Δ={final_delta:.3f}",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig



def _render_run_banner(summary: dict[str, Any]) -> str:
    epochs = int(summary.get("epochs_count") or 0)
    one_epoch_note = "One-epoch run (not truncation)." if epochs == 1 else f"Epoch depth: {epochs}."
    return (
        f"Run selection: {summary.get('run_id')} | "
        f"{one_epoch_note} "
        f"Accepted epochs: {summary.get('accepted_prompt_updates')}/{epochs}"
    )



def _save_html_report(
    html_path: Path,
    png_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    image_base64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_banner = _render_run_banner(summary)

    table_rows = "\n".join(
        "<tr>"
        f"<td>{int(row['epoch'])}</td>"
        f"<td>{row['winner']}</td>"
        f"<td>{row['avg_score_a']:.3f}</td>"
        f"<td>{row['avg_score_b']:.3f}</td>"
        f"<td>{row['improvement_delta']:.3f}</td>"
        f"<td>{'yes' if row.get('quality_gate_passed') else 'no'}</td>"
        f"<td>{'yes' if row.get('runtime_gate_passed') else 'no'}</td>"
        f"<td>{'yes' if row.get('stability_gate_passed') else 'no'}</td>"
        f"<td>{int(row.get('timeout_count') or 0)}</td>"
        "</tr>"
        for row in rows
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Prompt Improvement Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; background: #0f1115; color: #e8ecf1; }}
        h1, h2 {{ margin: 0 0 12px 0; }}
        .muted {{ color: #a7b0bc; }}
        .banner {{ margin: 14px 0; padding: 12px; border-radius: 8px; border: 1px solid #2a3240; background: #151b25; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 14px 0 18px 0; }}
        .card {{ background: #171b22; border: 1px solid #2a3240; border-radius: 10px; padding: 12px; }}
        .card .label {{ color: #9aa5b4; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
        .card .value {{ font-size: 22px; margin-top: 6px; font-weight: 600; }}
        .dash {{ width: 100%; border-radius: 10px; border: 1px solid #2a3240; background: #fff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
        th, td {{ border: 1px solid #2a3240; padding: 8px 10px; font-size: 13px; text-align: left; }}
        th {{ background: #171b22; }}
        tr:nth-child(even) td {{ background: #12161d; }}
        .footer {{ margin-top: 16px; color: #95a0af; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Prompt Improvement Dashboard (V2)</h1>
    <p class=\"muted\">Run ID: {summary['run_id']} | Generated: {generated_at} | Quality score: {summary['quality_score']:.2f}</p>
    <div class=\"banner\">{run_banner}</div>

    <div class=\"grid\">
        <div class=\"card\"><div class=\"label\">Epochs</div><div class=\"value\">{summary['epochs_count']}</div></div>
        <div class=\"card\"><div class=\"label\">Improved Epochs</div><div class=\"value\">{summary['improved_epochs']}</div></div>
        <div class=\"card\"><div class=\"label\">Accepted Epochs</div><div class=\"value\">{summary['accepted_prompt_updates']}</div></div>
        <div class=\"card\"><div class=\"label\">Net Δ</div><div class=\"value\">{summary['net_improvement_delta']:.3f}</div></div>
    </div>

    <img class=\"dash\" src=\"data:image/png;base64,{image_base64}\" alt=\"Prompt improvement dashboard\" />

    <h2 style=\"margin-top:20px\">Epoch Gate Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Epoch</th>
                <th>Winner</th>
                <th>Avg A</th>
                <th>Avg B</th>
                <th>Δ (B-A)</th>
                <th>Quality</th>
                <th>Runtime</th>
                <th>Stability</th>
                <th>Timeouts (B)</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <div class=\"footer\">Data source: {summary['history_path']}</div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")



def _render_single_run(
    run_id: str,
    selected_records: list[dict[str, Any]],
    input_path: str,
    output_dir: Path,
    base_name: str,
) -> VisualizationResult:
    epochs = _epoch_records(selected_records)
    if not epochs:
        raise ValueError(f"Run {run_id} has no epoch records to visualize")

    rows = _build_rows(selected_records)
    if not rows:
        raise ValueError(f"Run {run_id} produced no usable rows for visualization")

    block_ids = _collect_block_ids(epochs)
    block_matrix, matrix_epochs = _build_block_delta_matrix(epochs, block_ids)

    file_stem = f"{base_name}_{run_id[:8]}"
    png_path = output_dir / f"{file_stem}.png"
    html_path = output_dir / f"{file_stem}.html"
    summary_path = output_dir / f"{file_stem}.summary.json"

    quality_score = _run_quality_score(rows)
    fig = _plot_dashboard(rows, block_ids, block_matrix, matrix_epochs, run_id=run_id, quality_score=quality_score)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    improved_epochs = sum(1 for row in rows if _safe_float(row.get("improvement_delta"), 0.0) > 0)
    accepted_prompt_updates = sum(1 for row in rows if str(row.get("winner", "")) == "candidate")
    net_improvement_delta = sum(_safe_float(row.get("improvement_delta"), 0.0) for row in rows)

    summary = {
        "run_id": run_id,
        "history_path": str(Path(input_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "epochs_count": len(rows),
        "improved_epochs": improved_epochs,
        "accepted_prompt_updates": accepted_prompt_updates,
        "net_improvement_delta": net_improvement_delta,
        "quality_score": quality_score,
        "png_path": str(png_path.resolve()),
        "html_path": str(html_path.resolve()),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_html_report(html_path=html_path, png_path=png_path, summary=summary, rows=rows)

    return VisualizationResult(
        run_id=run_id,
        history_path=str(Path(input_path).resolve()),
        output_dir=str(output_dir.resolve()),
        png_path=str(png_path.resolve()),
        html_path=str(html_path.resolve()),
        summary_path=str(summary_path.resolve()),
        epochs_count=len(rows),
        improved_epochs=improved_epochs,
        accepted_prompt_updates=accepted_prompt_updates,
        net_improvement_delta=net_improvement_delta,
        quality_score=quality_score,
    )



def _render_golden_alias(golden_run: VisualizationResult, output_dir: Path) -> tuple[str, str]:
    golden_png_path = output_dir / "golden_run.png"
    golden_html_path = output_dir / "golden_run.html"

    source_png = Path(golden_run.png_path)
    source_html = Path(golden_run.html_path)
    golden_png_path.write_bytes(source_png.read_bytes())

    html_text = source_html.read_text(encoding="utf-8")
    html_text = html_text.replace("Prompt Improvement Dashboard", "Golden Run Dashboard")
    html_text = html_text.replace(
        f"Run ID: {golden_run.run_id}",
        f"Run ID: {golden_run.run_id} | GOLDEN RUN",
    )
    golden_html_path.write_text(html_text, encoding="utf-8")
    return str(golden_png_path.resolve()), str(golden_html_path.resolve())



def _render_index_report(
    *,
    output_dir: Path,
    runs: list[VisualizationResult],
    selected_mode: str,
) -> tuple[str, str]:
    index_json_path = output_dir / "prompt_improvement_index.json"
    index_html_path = output_dir / "prompt_improvement_index.html"

    ordered = sorted(runs, key=lambda item: item.quality_score, reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_mode": selected_mode,
        "runs_count": len(ordered),
        "runs": [run.to_dict() for run in ordered],
    }
    index_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows_html = "\n".join(
        "<tr>"
        f"<td>{idx + 1}</td>"
        f"<td>{run.run_id}</td>"
        f"<td>{run.epochs_count}</td>"
        f"<td>{run.accepted_prompt_updates}</td>"
        f"<td>{run.net_improvement_delta:.3f}</td>"
        f"<td>{run.quality_score:.2f}</td>"
        f"<td><a href='{Path(run.html_path).name}'>dashboard</a></td>"
        "</tr>"
        for idx, run in enumerate(ordered)
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Prompt Improvement Index</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; background: #0f1115; color: #e8ecf1; }}
        h1 {{ margin: 0 0 8px 0; }}
        .muted {{ color: #a7b0bc; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
        th, td {{ border: 1px solid #2a3240; padding: 8px 10px; font-size: 13px; text-align: left; }}
        th {{ background: #171b22; }}
        tr:nth-child(even) td {{ background: #12161d; }}
        a {{ color: #8ec5ff; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>Prompt Improvement Index</h1>
    <p class=\"muted\">Selection mode: {selected_mode} | Runs rendered: {len(ordered)}</p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Run ID</th>
                <th>Epochs</th>
                <th>Accepted Epochs</th>
                <th>Net Δ</th>
                <th>Quality Score</th>
                <th>Dashboard</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>
"""
    index_html_path.write_text(html, encoding="utf-8")

    return str(index_json_path.resolve()), str(index_html_path.resolve())



def create_prompt_improvement_visualization(
    input_path: str = "data/prompt_suite_generations.json",
    output_dir: str = "plots/prompt_improvement",
    run_id: str | None = None,
    base_name: str = "prompt_improvement_dashboard",
    latest_run: bool = True,
    emit_index: bool = True,
    golden_alias: bool = False,
    legacy_golden_alias: bool = False,
) -> VisualizationBatchResult:
    if not ENABLE_VISUALIZER_V2:
        latest_run = False if not run_id else latest_run
        emit_index = False
        if not golden_alias and not legacy_golden_alias:
            golden_alias = True

    history_records = _load_generations(input_path)
    if not history_records:
        raise ValueError(f"No generation data found in {input_path}")

    runs_map = _records_by_run(history_records)
    if not runs_map:
        raise ValueError(f"No run_id entries found in {input_path}")

    if run_id:
        run_ids = [run_id]
        selected_mode = "explicit_run_id"
    else:
        ordered_ids = sorted(
            runs_map.keys(),
            key=lambda rid: max(float(r.get("timestamp") or 0.0) for r in runs_map[rid]),
        )
        if latest_run:
            run_ids = ordered_ids[-1:]
            selected_mode = "latest_run"
        else:
            run_ids = ordered_ids
            selected_mode = "all_runs"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[VisualizationResult] = []
    for selected_run_id in run_ids:
        run_records = runs_map.get(selected_run_id)
        if not run_records:
            continue
        try:
            run_results.append(
                _render_single_run(
                    run_id=selected_run_id,
                    selected_records=run_records,
                    input_path=input_path,
                    output_dir=out_dir,
                    base_name=base_name,
                )
            )
        except ValueError as exc:
            if "no epoch records" in str(exc).lower():
                continue
            raise

    if not run_results:
        raise ValueError("No eligible runs were rendered")

    golden_run = max(run_results, key=lambda item: item.quality_score)
    golden_png_path = ""
    golden_html_path = ""
    if bool(golden_alias or legacy_golden_alias):
        golden_png_path, golden_html_path = _render_golden_alias(golden_run=golden_run, output_dir=out_dir)

    index_json_path = None
    index_html_path = None
    if emit_index:
        index_json_path, index_html_path = _render_index_report(
            output_dir=out_dir,
            runs=run_results,
            selected_mode=selected_mode,
        )

    return VisualizationBatchResult(
        history_path=str(Path(input_path).resolve()),
        output_dir=str(out_dir.resolve()),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runs=run_results,
        golden_run_id=golden_run.run_id,
        golden_png_path=golden_png_path,
        golden_html_path=golden_html_path,
        index_json_path=index_json_path,
        index_html_path=index_html_path,
    )
