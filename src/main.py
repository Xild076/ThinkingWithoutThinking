from __future__ import annotations

import argparse
import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path


def _import_pipeline():
    try:
        from pipeline import Pipeline
    except Exception:
        from src.pipeline import Pipeline
    return Pipeline


def _import_prompt_training():
    try:
        from prompt_training import PromptSuiteStore, run_training_loop
    except Exception:
        from src.prompt_training import PromptSuiteStore, run_training_loop
    return PromptSuiteStore, run_training_loop


def _import_app():
    try:
        from app import app
    except Exception:
        from src.app import app
    return app


def _import_visualizer():
    try:
        from prompt_improvement_visualizer import create_prompt_improvement_visualization
    except Exception:
        from src.prompt_improvement_visualizer import create_prompt_improvement_visualization
    return create_prompt_improvement_visualization


def _resolve_app_import_string() -> str:
    candidates = ["src.app:app", "app:app"]
    for candidate in candidates:
        module_name, _ = candidate.split(":", 1)
        try:
            importlib.import_module(module_name)
            return candidate
        except Exception:
            continue
    return "src.app:app"


def _discover_lan_ip() -> str | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        finally:
            sock.close()
        if ip and ip != "127.0.0.1":
            return ip
    except Exception:
        return None
    return None


def cmd_run(args: argparse.Namespace) -> int:
    Pipeline = _import_pipeline()
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Prompt is required via --prompt or stdin", file=sys.stderr)
        return 2

    pipeline = Pipeline()
    result = pipeline.run(prompt=prompt, thinking_level=args.thinking)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result.get("response", ""))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving API/UI on {args.host}:{args.port}")
    print(f"Local URL:   http://127.0.0.1:{args.port}")
    if args.host in {"0.0.0.0", "::"}:
        lan_ip = _discover_lan_ip()
        if lan_ip:
            print(f"Network URL: http://{lan_ip}:{args.port}")

    uvicorn.run(_resolve_app_import_string(), host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_streamlit(args: argparse.Namespace) -> int:
    print(f"Serving Streamlit on {args.host}:{args.port}")
    print(f"Local URL:   http://127.0.0.1:{args.port}")
    if args.host in {"0.0.0.0", "::"}:
        lan_ip = _discover_lan_ip()
        if lan_ip:
            print(f"Network URL: http://{lan_ip}:{args.port}")

    cmd = [
        "streamlit",
        "run",
        "src/streamlit_app.py",
        "--server.address",
        str(args.host),
        "--server.port",
        str(args.port),
    ]
    return subprocess.call(cmd)


def cmd_train(args: argparse.Namespace) -> int:
    _, run_training_loop = _import_prompt_training()
    result = run_training_loop(
        base_prompts_path=args.prompts,
        output_path=args.output,
        test_cases_path=args.dataset,
        epochs=args.epochs,
        num_test_cases_per_trial=args.train_sample_size,
        holdout_sample_size=args.holdout_sample_size,
        rca_case_budget=args.rca_case_budget,
        mutation_block_budget=args.mutation_block_budget,
        mutation_retry_enabled=args.mutation_retry_enabled,
        mutation_max_retries=args.mutation_max_retries,
        generalizer_cadence=args.generalizer_cadence,
        generalizer_suspicious_delta_threshold=args.generalizer_suspicious_delta_threshold,
        bootstrap_resamples=args.bootstrap_resamples,
        holdout_split_ratio=args.holdout_split_ratio,
        random_seed=args.seed,
        thinking_level=args.thinking,
        progress_status_path=args.progress_status,
        progress_events_path=args.progress_events,
        progress_events_archive_dir=args.progress_events_archive_dir,
        track_progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_generations(args: argparse.Namespace) -> int:
    PromptSuiteStore, _ = _import_prompt_training()
    store = PromptSuiteStore(args.output)
    records = store.list_generations()
    print(json.dumps(records, indent=2))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    PromptSuiteStore, _ = _import_prompt_training()
    store = PromptSuiteStore(args.output)
    prompts = store.get_generation(args.generation, run_id=args.run_id)
    if prompts is None:
        if args.run_id:
            print(f"Generation {args.generation} for run_id {args.run_id} not found", file=sys.stderr)
        else:
            print(f"Generation {args.generation} not found", file=sys.stderr)
        return 1

    Path(args.prompts).write_text(json.dumps(prompts, indent=2))
    print(
        json.dumps(
            {
                "status": "ok",
                "generation": args.generation,
                "run_id": args.run_id or "latest",
                "saved_to": args.prompts,
            },
            indent=2,
        )
    )
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    create_prompt_improvement_visualization = _import_visualizer()
    batch_result = create_prompt_improvement_visualization(
        input_path=args.input,
        output_dir=args.output_dir,
        run_id=args.run_id or None,
        base_name=args.base_name,
        latest_run=bool(args.latest_run),
        emit_index=bool(args.emit_index),
        golden_alias=bool(args.golden_alias),
    )
    payload = batch_result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Runs visualized: {payload['runs_count']}")
        print(f"Golden run:      {payload['golden_run_id']}")
        if payload.get("golden_png_path"):
            print(f"Golden PNG:      {payload['golden_png_path']}")
            print(f"Golden HTML:     {payload['golden_html_path']}")
        if payload.get("index_html_path"):
            print(f"Index HTML:      {payload['index_html_path']}")
        if payload.get("index_json_path"):
            print(f"Index JSON:      {payload['index_json_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinking",
        description="Centralized CLI for pipeline, UIs, and prompt training",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline once")
    run_p.add_argument("--prompt", type=str, default="", help="Prompt text")
    run_p.add_argument(
        "--thinking",
        type=str,
        default="med-synth",
        choices=["low", "med-synth", "med-plan", "high"],
    )
    run_p.add_argument("--json", action="store_true", help="Print full JSON output")
    run_p.add_argument("--output", type=str, default="", help="Optional JSON file output path")
    run_p.set_defaults(func=cmd_run)

    serve_p = sub.add_parser("serve", help="Serve browser UI + API")
    serve_p.add_argument("--host", type=str, default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")
    serve_p.set_defaults(func=cmd_serve)

    streamlit_p = sub.add_parser("streamlit", help="Launch Streamlit UI")
    streamlit_p.add_argument("--host", type=str, default="0.0.0.0")
    streamlit_p.add_argument("--port", type=int, default=8501)
    streamlit_p.set_defaults(func=cmd_streamlit)

    train_p = sub.add_parser("train", help="Run prompt A/B training")
    train_p.add_argument("--prompts", type=str, default="data/prompts.json")
    train_p.add_argument("--output", type=str, default="data/prompt_suite_generations.json")
    train_p.add_argument("--dataset", type=str, default="data/prompt_train_cases.json")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--train-sample-size", type=int, default=6)
    train_p.add_argument("--sample-size", dest="train_sample_size", type=int, help=argparse.SUPPRESS)
    train_p.add_argument("--holdout-sample-size", type=int, default=6)
    train_p.add_argument("--rca-case-budget", type=int, default=3)
    train_p.add_argument("--mutation-block-budget", type=int, default=3)
    train_p.add_argument("--mutation-retry-enabled", action=argparse.BooleanOptionalAction, default=True)
    train_p.add_argument("--mutation-max-retries", type=int, default=3)
    train_p.add_argument("--generalizer-cadence", type=int, default=3)
    train_p.add_argument("--generalizer-suspicious-delta-threshold", type=int, default=3)
    train_p.add_argument("--bootstrap-resamples", type=int, default=1000)
    train_p.add_argument("--holdout-split-ratio", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--progress-status", type=str, default="data/training_status.json")
    train_p.add_argument("--progress-events", type=str, default="data/training_events.jsonl")
    train_p.add_argument("--progress-events-archive-dir", type=str, default="data/training_events")
    train_p.add_argument("--no-progress", action="store_true", help="Disable structured progress tracking files")
    train_p.add_argument(
        "--thinking",
        type=str,
        default="med-synth",
        choices=["low", "med-synth", "med-plan", "high"],
    )
    train_p.set_defaults(func=cmd_train)

    gen_p = sub.add_parser("generations", help="List stored prompt generations")
    gen_p.add_argument("--output", type=str, default="data/prompt_suite_generations.json")
    gen_p.set_defaults(func=cmd_generations)

    rollback_p = sub.add_parser("rollback", help="Rollback prompts to a generation")
    rollback_p.add_argument("--output", type=str, default="data/prompt_suite_generations.json")
    rollback_p.add_argument("--prompts", type=str, default="data/prompts.json")
    rollback_p.add_argument("--generation", type=int, required=True)
    rollback_p.add_argument("--run-id", type=str, default="", help="Optional run_id to disambiguate repeated generations")
    rollback_p.set_defaults(func=cmd_rollback)

    vis_p = sub.add_parser("visualize", help="Visualize prompt evolution and improvement over training epochs")
    vis_p.add_argument("--input", type=str, default="data/prompt_suite_generations.json")
    vis_p.add_argument("--output-dir", type=str, default="plots/prompt_improvement")
    vis_p.add_argument("--run-id", type=str, default="", help="Optional run_id (defaults to all runs)")
    vis_p.add_argument("--base-name", type=str, default="prompt_improvement_dashboard")
    vis_p.add_argument("--latest-run", action=argparse.BooleanOptionalAction, default=True)
    vis_p.add_argument("--emit-index", action=argparse.BooleanOptionalAction, default=True)
    vis_p.add_argument("--golden-alias", action=argparse.BooleanOptionalAction, default=False)
    vis_p.add_argument("--json", action="store_true", help="Print result as JSON")
    vis_p.set_defaults(func=cmd_visualize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
