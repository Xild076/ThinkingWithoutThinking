# Branch Comparison: selfimprovement vs main

**Date:** February 17, 2026  
**Comparison:** `selfimprovement` (ad2b2d3) vs `main` (21f8ee0)

## Executive Summary

The `selfimprovement` branch represents a **complete architectural overhaul** of the ThinkingWithoutThinking project. What started as a Streamlit-based prompt engineering experiment has been transformed into a sophisticated System-2 prompting pipeline with self-optimization capabilities.

### Scale of Changes
- **651 files changed** (+73,202 insertions, -5,123 deletions)
- **627 files added**
- **10 files deleted**
- **12 files modified**
- **Single commit** on selfimprovement branch: "Add initial implementation of telemetry ingestion architecture and fault tolerance features"

## Architecture Comparison

### Main Branch (Current Production)
**Purpose:** Simple prompt engineering experiment with Streamlit UI

**Core Components:**
- `src/thinking_pipeline.py` - Basic thinking pipeline
- `src/pipeline_blocks.py` - Pipeline block definitions
- `src/utility.py` - Utility functions
- `app.py` - Streamlit application
- Uses Google Gemma 27B model
- Simple self-critique mechanism

**Key Features:**
- Multi-step reasoning with structured thinking blocks
- Executable code tool (sandboxed Python)
- Live web search
- Self-correction feedback loops
- Streamlit UI for interactive tracing
- 4-phase process: Plan & Route → Execute → Synthesize & Refine → Present

### Selfimprovement Branch (Proposed)
**Purpose:** Experimental System-2 style prompting pipeline + prompt self-optimization loop

**Core Components:**
- `src/pipeline.py` - Advanced pipeline orchestration
- `src/pipeline_blocks.py` - Modular block definitions (completely rewritten)
- `src/prompt_training.py` - **NEW**: Prompt optimization/training loop
- `src/app.py` - **NEW**: FastAPI server for UI + API + training stream
- `src/main.py` - **NEW**: Centralized CLI entrypoint
- `src/rate_limit_monitoring.py` - **NEW**: Rate limiting and monitoring
- `src/streamlit_app.py` - **NEW**: Streamlit integration (separate from FastAPI)
- `src/ui/index.html` - **NEW**: Pipeline run dashboard
- `src/ui/training.html` - **NEW**: Training monitor dashboard
- `src/tools/` directory - **NEW**: Organized tool implementations
  - `python_exec_tool.py` - Python execution
  - `web_search_tool.py` - Web search functionality
  - `wikipedia_tool.py` - Wikipedia integration

**Key Features:**
1. **Modular inference pipeline** with planning, routing, tools, synthesis, critique
2. **Training loop** that A/B tests prompt suites, scores outcomes, runs root-cause analysis
3. **Iterative prompt improvement** through automated testing
4. **FastAPI server** with multiple endpoints
5. **Dual UI system**: Browser-based dashboards + Streamlit
6. **CLI tool** with multiple run modes
7. **Telemetry ingestion architecture**
8. **Fault tolerance features**
9. **Training monitoring** via SSE (Server-Sent Events)
10. **Prompt versioning** and rollback capabilities

## File Structure Differences

### Main Branch Structure
```
├── app.py                          # Streamlit app
├── src/
│   ├── thinking_pipeline.py       # Core pipeline
│   ├── pipeline_blocks.py         # Block definitions
│   ├── pdf_utils.py               # PDF utilities
│   └── utility.py                 # Helper functions
├── tests/                          # Unit tests
├── requirements.txt
└── README.md
```

### Selfimprovement Branch Structure
```
├── src/
│   ├── app.py                     # FastAPI server (NEW)
│   ├── main.py                    # CLI entrypoint (NEW)
│   ├── pipeline.py                # Pipeline orchestration (REWRITTEN)
│   ├── pipeline_blocks.py         # Modular blocks (REWRITTEN)
│   ├── prompt_training.py         # Training loop (NEW)
│   ├── streamlit_app.py           # Streamlit UI (NEW)
│   ├── rate_limit_monitoring.py   # Monitoring (NEW)
│   ├── utility.py                 # Utilities (MODIFIED)
│   ├── ui/                        # Browser UIs (NEW)
│   │   ├── index.html
│   │   └── training.html
│   └── tools/                     # Tool modules (NEW)
│       ├── python_exec_tool.py
│       ├── web_search_tool.py
│       └── wikipedia_tool.py
├── data/                          # Training data (NEW)
│   ├── prompts.json
│   ├── prompt_train_cases.json
│   ├── prompt_suite_generations.json
│   ├── training_status.json
│   ├── training_events.jsonl
│   └── rates.csv
├── logs/                          # Application logs (NEW)
│   ├── pipeline.log
│   └── prompt_training.log
├── temp/                          # Temporary execution files (NEW)
├── PLAN.MD                        # Development plan (NEW)
├── writeup.md                     # Documentation (NEW)
└── requirements.txt               # Updated dependencies
```

## Feature Comparison

| Feature | Main Branch | Selfimprovement Branch |
|---------|-------------|------------------------|
| **UI Framework** | Streamlit only | FastAPI + HTML + Streamlit |
| **API Server** | ❌ None | ✅ FastAPI with REST endpoints |
| **CLI Tool** | ❌ None | ✅ Full CLI with subcommands |
| **Prompt Training** | ❌ None | ✅ Automated A/B testing |
| **Training Monitoring** | ❌ None | ✅ Real-time dashboard + SSE |
| **Prompt Versioning** | ❌ None | ✅ Generation tracking + rollback |
| **Tool Organization** | ❌ Monolithic | ✅ Modular tools/ directory |
| **Rate Limiting** | ❌ None | ✅ Monitoring and tracking |
| **Telemetry** | ❌ None | ✅ Ingestion architecture |
| **Fault Tolerance** | ❌ Basic | ✅ Enhanced |
| **Logging** | ❌ Minimal | ✅ Structured logs/ directory |
| **Data Management** | ❌ None | ✅ Structured data/ directory |
| **Test Framework** | ✅ pytest | ✅ pytest (enhanced) |

## API Endpoints (Selfimprovement Only)

The selfimprovement branch introduces a complete REST API:

- `GET /health` - Health check
- `POST /run` - Execute pipeline
- `POST /reload-prompts` - Reload prompt configurations
- `GET /training/status` - Get current training status
- `GET /training/events` - Fetch training events
- `GET /training/stream` - SSE stream for real-time updates

## CLI Commands (Selfimprovement Only)

### Serve Mode
```bash
python -m src.main serve --host 0.0.0.0 --port 8000 --reload
```

### Run Mode
```bash
python -m src.main run --prompt "..." --thinking med-synth --json
```

### Train Mode
```bash
python -m src.main train \
    --prompts data/prompts.json \
    --dataset data/prompt_train_cases.json \
    --output data/prompt_suite_generations.json \
    --epochs 10 \
    --sample-size 5 \
    --thinking med-synth
```

### Utility Commands
```bash
# List stored generations
python -m src.main generations --output data/prompt_suite_generations.json

# Rollback prompts
python -m src.main rollback --generation 3 --prompts data/prompts.json
```

## Training System (Selfimprovement Only)

The selfimprovement branch introduces a sophisticated prompt training system:

### Training Loop Components
1. **A/B Testing**: Tests multiple prompt suites against test cases
2. **Scoring**: Evaluates outcomes based on predefined metrics
3. **Root-Cause Analysis**: Identifies why certain prompts perform better
4. **Iterative Improvement**: Automatically refines prompts based on analysis

### Training Artifacts
- `data/training_status.json` - Latest status snapshot
- `data/training_events.jsonl` - Append-only event stream
- `logs/prompt_training.log` - Textual run log

### Training Dashboard Features
- Real-time progress monitoring
- Current case metadata (id, category, difficulty, preview, timing)
- Heartbeat events every ~3s when idle
- Default stream poll interval: 500ms

## Thinking Levels

Both branches support thinking levels, but selfimprovement formalizes them:

- `low` - Minimal reasoning
- `med-synth` - Medium synthesis
- `med-plan` - Medium planning
- `high` - Maximum reasoning depth

## Dependencies

Both branches use similar base dependencies, but selfimprovement adds:
- **FastAPI** - Web framework
- **Server-Sent Events** support
- Enhanced **asyncio** capabilities
- Additional data processing libraries

## Backwards Compatibility

⚠️ **Breaking Changes**: The selfimprovement branch is NOT backwards compatible with main.

- Different entry points (`src/main.py` vs `app.py`)
- Different architecture (FastAPI vs Streamlit-only)
- Different configuration format
- Different data structures

## Migration Path

To migrate from main to selfimprovement:

1. **Data Migration**: Export any existing prompts/configurations
2. **Code Migration**: Rewrite integrations to use new API
3. **Configuration**: Update environment variables and configs
4. **Testing**: Thoroughly test all pipelines and tools
5. **Training**: Set up training data in new format
6. **Monitoring**: Configure new logging and telemetry systems

## Recommendations

### For Production Use
- **Main branch** is suitable for:
  - Simple prompt engineering experiments
  - Quick prototypes
  - Streamlit-focused applications
  - Teams familiar with Streamlit

- **Selfimprovement branch** is suitable for:
  - Advanced prompt engineering research
  - Automated prompt optimization
  - Production API deployments
  - Teams needing advanced monitoring and telemetry
  - Projects requiring prompt versioning and rollback
  - Systems requiring fault tolerance

### Development Status

Both branches are marked as "actively experimental":
- **Main**: Stable experimental prototype
- **Selfimprovement**: Comprehensive rewrite with production-grade features but still experimental

### Testing

- **Main**: Basic pytest suite
- **Selfimprovement**: Enhanced test infrastructure recommended for production use

## Commit History

### Main Branch (Recent Commits)
- 21f8ee0 - Refactor link retrieval in Google News integration
- a10939f - Fix requirements files by removing duplicate reportlab entry
- e398fe7 - Adds PDF support and advanced pipeline blocks
- 7003342 - Enhance pipeline functionality with new MathImprovementBlock
- a76c8db - Small updates
- 90df18a - Add unit tests for pipeline blocks
- 0ff07f6 - Improved prompt engineering
- 6cd3d20 - Added more personality

### Selfimprovement Branch
- ad2b2d3 - Add initial implementation of telemetry ingestion architecture and fault tolerance features

## Conclusion

The selfimprovement branch represents a fundamental redesign of ThinkingWithoutThinking:

**Pros:**
- ✅ More professional architecture
- ✅ Automated prompt optimization
- ✅ Production-ready API
- ✅ Advanced monitoring and telemetry
- ✅ Better code organization
- ✅ Comprehensive CLI tooling
- ✅ Prompt versioning and rollback
- ✅ Enhanced fault tolerance

**Cons:**
- ❌ Complete rewrite requires learning new architecture
- ❌ Not backwards compatible
- ❌ More complex to set up and maintain
- ❌ Requires migration of existing work
- ❌ Single massive commit makes incremental review difficult

**Recommendation:** The selfimprovement branch should be considered for adoption if:
1. You need production-grade features
2. You want automated prompt optimization
3. You require API access for integrations
4. You need advanced monitoring capabilities

However, a more incremental migration path would be beneficial for teams with existing main branch deployments.

## Concerns and Issues

### 1. Committed Temporary Files
The selfimprovement branch commits **hundreds of temporary code execution files** in `temp/code_exec/`:
- 300+ temporary Python execution files
- 300+ result.json files
- These should typically be in `.gitignore`

**Impact:** 
- Bloats repository size
- Makes diffs difficult to review
- May contain sensitive data from test runs
- Violates typical Git best practices

**Recommendation:** Add `temp/` to `.gitignore` and remove these files from Git history.

### 2. Committed Python Cache Files
The branch commits `__pycache__` directories with `.pyc` files:
- Should be in `.gitignore`
- Makes repository messy
- Can cause issues across Python versions

**Recommendation:** Add `__pycache__/` and `*.pyc` to `.gitignore` and clean from repository.

### 3. Single Massive Commit
The entire rewrite is in one commit (ad2b2d3):
- Difficult to review changes incrementally
- Hard to identify when specific features were added
- Makes bisecting issues nearly impossible
- Loses development history

**Recommendation:** Consider breaking down into logical commits if re-doing this work.

### 4. Dependency Management
New dependencies added without version pinning:
```
fastapi
openai
groq
uvicorn
```

**Recommendation:** Pin specific versions for reproducibility:
```
fastapi==0.115.0
openai==1.54.0
groq==0.11.0
uvicorn==0.32.0
```

### 5. Missing Documentation
While PLAN.MD and writeup.md exist, there's limited documentation on:
- Migration guide from main to selfimprovement
- API authentication and security
- Training data format specifications
- Deployment instructions

### 6. Testing Coverage
No evidence of updated tests for new features:
- No tests for FastAPI endpoints
- No tests for training loop
- No tests for new tools
- Tests directory not visible in selfimprovement branch structure

**Note:** Tests may exist but weren't visible in the file listing.
