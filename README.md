# Thinking Without Thinking (TWT)

Thinking Without Thinking is an experimental System-2 style prompting pipeline plus a prompt self-optimization loop.

The project has two major parts:

1. A modular inference pipeline (planning, routing, tools, synthesis, critique).
2. A training loop that A/B tests prompt suites, scores outcomes, runs root-cause analysis, and iteratively improves prompts.

## Repository Layout

- `src/pipeline.py` — pipeline orchestration.
- `src/pipeline_blocks.py` — modular block definitions and behavior.
- `src/prompt_training.py` — prompt optimization/training loop.
- `src/app.py` — FastAPI server for UI + API + training stream.
- `src/main.py` — centralized CLI entrypoint.
- `src/ui/index.html` — pipeline run dashboard.
- `src/ui/training.html` — training monitor dashboard.

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file (or export env vars) for any model/API credentials used by your configured tools and pipeline blocks.

## Run Modes

### A) Serve API + browser UIs

```bash
python -m src.main serve --host 0.0.0.0 --port 8000 --reload
```

Open:

- Pipeline UI: `http://127.0.0.1:8000/`
- Training UI: `http://127.0.0.1:8000/training`

### B) Run one pipeline request from CLI

```bash
python -m src.main run --prompt "Explain chain-of-thought safety at a high level" --thinking med-synth --json
```

Thinking levels:

- `low`
- `med-synth`
- `med-plan`
- `high`

### C) Run prompt training

```bash
python -m src.main train \
	--prompts data/prompts.json \
	--dataset data/prompt_train_cases.json \
	--output data/prompt_suite_generations.json \
	--epochs 10 \
	--sample-size 5 \
	--thinking med-synth
```

Training progress artifacts:

- `data/training_status.json` — latest status snapshot.
- `data/training_events.jsonl` — append-only event stream.
- `logs/prompt_training.log` — textual run log.

## Training UI Notes

The training dashboard consumes both polling endpoints and SSE:

- `GET /training/status`
- `GET /training/events`
- `GET /training/stream?poll_ms=...`

Recent stream behavior:

- Default stream poll interval is `500ms`.
- Heartbeat events emit roughly every `3s` when idle.
- UI supports richer current-case metadata from training events/status (`id`, `category`, `difficulty`, preview, timing).

## API Endpoints (Core)

- `GET /health`
- `POST /run`
- `POST /reload-prompts`
- `GET /training/status`
- `GET /training/events`
- `GET /training/stream`

## Useful CLI Commands

List stored generations:

```bash
python -m src.main generations --output data/prompt_suite_generations.json
```

Rollback prompts to a stored generation:

```bash
python -m src.main rollback --generation 3 --prompts data/prompts.json
```

## Development

Run tests:

```bash
pytest -q
```

If a test run fails due to missing credentials or optional external tools, run targeted tests for the module you changed.

## Status

This project is actively experimental and intended for rapid iteration and research-style exploration.

