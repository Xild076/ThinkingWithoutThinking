# Visual Architecture Comparison

## Main Branch Architecture

```
┌─────────────────────────────────────────────────┐
│                                                 │
│              User Interface                     │
│            (Streamlit App)                      │
│                 app.py                          │
│                                                 │
└────────────────┬────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│           Thinking Pipeline                     │
│         src/thinking_pipeline.py                │
│                                                 │
│   ┌──────────────────────────────────────┐     │
│   │  Phase 1: Plan & Route               │     │
│   │  - Generate high-level strategy      │     │
│   │  - Critique and refine               │     │
│   │  - Create block sequence             │     │
│   └──────────────────────────────────────┘     │
│                                                 │
│   ┌──────────────────────────────────────┐     │
│   │  Phase 2: Execute                    │     │
│   │  - Internet Tool                     │     │
│   │  - Code Tool (sandboxed Python)      │     │
│   │  - Creative Idea Tool                │     │
│   └──────────────────────────────────────┘     │
│                                                 │
│   ┌──────────────────────────────────────┐     │
│   │  Phase 3: Synthesize & Refine        │     │
│   │  - Gather context                    │     │
│   │  - Generate answer                   │     │
│   │  - Score & improve                   │     │
│   └──────────────────────────────────────┘     │
│                                                 │
│   ┌──────────────────────────────────────┐     │
│   │  Phase 4: Present                    │     │
│   │  - Format output                     │     │
│   │  - Display in UI                     │     │
│   └──────────────────────────────────────┘     │
│                                                 │
└─────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│          Pipeline Blocks                        │
│       src/pipeline_blocks.py                    │
│                                                 │
│  - Block definitions                            │
│  - Tool implementations                         │
│  - Utilities                                    │
└─────────────────────────────────────────────────┘
```

**Key Characteristics:**
- ✅ Simple, monolithic architecture
- ✅ Single entry point (Streamlit)
- ✅ All-in-one approach
- ✅ Easy to understand
- ❌ No API access
- ❌ No automation
- ❌ Limited scalability

---

## Selfimprovement Branch Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interfaces                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Browser    │  │  Streamlit   │  │     CLI      │         │
│  │ UI Dashboard │  │     App      │  │   Commands   │         │
│  │ (index.html) │  │streamlit_app │  │  src/main.py │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          v                 v                 v
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server                               │
│                     src/app.py                                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   API Endpoints                          │  │
│  │                                                          │  │
│  │  GET  /health              - Health check               │  │
│  │  POST /run                 - Execute pipeline           │  │
│  │  POST /reload-prompts      - Reload configurations      │  │
│  │  GET  /training/status     - Training status            │  │
│  │  GET  /training/events     - Training events            │  │
│  │  GET  /training/stream     - SSE real-time updates      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           v                                  v
┌──────────────────────────────┐    ┌──────────────────────────┐
│    Pipeline Orchestrator     │    │   Training System        │
│      src/pipeline.py         │    │  src/prompt_training.py  │
│                              │    │                          │
│  ┌────────────────────────┐  │    │  ┌────────────────────┐ │
│  │  Planning              │  │    │  │  A/B Testing       │ │
│  │  - Generate plans      │  │    │  │  - Test suites     │ │
│  │  - Route to tools      │  │    │  │  - Score outcomes  │ │
│  └────────────────────────┘  │    │  └────────────────────┘ │
│                              │    │                          │
│  ┌────────────────────────┐  │    │  ┌────────────────────┐ │
│  │  Tool Execution        │  │    │  │  Analysis          │ │
│  │  - Web search          │  │    │  │  - Root cause      │ │
│  │  - Code execution      │  │    │  │  - Improvements    │ │
│  │  - Wikipedia           │  │    │  └────────────────────┘ │
│  └────────────────────────┘  │    │                          │
│                              │    │  ┌────────────────────┐ │
│  ┌────────────────────────┐  │    │  │  Optimization      │ │
│  │  Synthesis             │  │    │  │  - Refine prompts  │ │
│  │  - Aggregate results   │  │    │  │  - Version control │ │
│  │  - Generate output     │  │    │  └────────────────────┘ │
│  └────────────────────────┘  │    │                          │
│                              │    └──────────────────────────┘
└──────────┬───────────────────┘                 │
           │                                     │
           v                                     v
┌──────────────────────────────┐    ┌──────────────────────────┐
│    Pipeline Blocks           │    │   Data Management        │
│  src/pipeline_blocks.py      │    │                          │
│                              │    │  ┌────────────────────┐  │
│  - Modular block defs        │    │  │  data/             │  │
│  - Extensible architecture   │    │  │  - prompts.json    │  │
│  - Tool integrations         │    │  │  - train_cases     │  │
└──────────┬───────────────────┘    │  │  - generations     │  │
           │                        │  │  - status          │  │
           v                        │  └────────────────────┘  │
┌──────────────────────────────┐    │                          │
│        Tool Modules          │    │  ┌────────────────────┐  │
│       src/tools/             │    │  │  logs/             │  │
│                              │    │  │  - pipeline.log    │  │
│  - python_exec_tool.py       │    │  │  - training.log    │  │
│  - web_search_tool.py        │    │  └────────────────────┘  │
│  - wikipedia_tool.py         │    └──────────────────────────┘
└──────────────────────────────┘
           │
           v
┌──────────────────────────────┐
│  Monitoring & Telemetry      │
│                              │
│  - rate_limit_monitoring.py  │
│  - Telemetry ingestion       │
│  - Fault tolerance           │
└──────────────────────────────┘
```

**Key Characteristics:**
- ✅ Multi-tier architecture
- ✅ Multiple entry points (Browser, CLI, Streamlit)
- ✅ RESTful API
- ✅ Automated training loop
- ✅ Comprehensive monitoring
- ✅ Modular tool system
- ✅ Data versioning
- ✅ Production-ready features
- ⚠️  More complex to maintain
- ⚠️  Steeper learning curve

---

## Data Flow Comparison

### Main Branch Data Flow
```
User Input (Streamlit)
    ↓
Thinking Pipeline
    ↓
Execute Blocks (in-memory)
    ↓
Synthesize Result
    ↓
Display in Streamlit
```

### Selfimprovement Branch Data Flow
```
User Input (Browser/CLI/Streamlit)
    ↓
FastAPI Router
    ↓
Pipeline Orchestrator
    ↓
Block Execution (modular tools)
    ↓
Results Storage (data/)
    ↓
Response (JSON/Stream)
    ↓
Display in UI

Parallel Process:
Training System
    ↓
A/B Test Prompts
    ↓
Score & Analyze
    ↓
Update Prompts (versioned)
    ↓
Store in data/
```

---

## Component Count

### Main Branch
```
Entry Points:      1 (Streamlit)
API Endpoints:     0
Core Modules:      3
Tool Modules:      0 (inline)
UI Files:          1
Data Files:        0 (runtime only)
Log Files:         0
Config Files:      1
```

### Selfimprovement Branch
```
Entry Points:      3 (Browser, CLI, Streamlit)
API Endpoints:     6
Core Modules:      7
Tool Modules:      3 (dedicated)
UI Files:          2
Data Files:        7 (persistent)
Log Files:         2 (persistent)
Config Files:      2
```

---

## Scalability Comparison

### Main Branch
```
Concurrent Users:   Limited by Streamlit
Deployment:         Single server
Horizontal Scale:   Difficult
API Integration:    Not possible
Automation:         Manual only
Monitoring:         Basic
```

### Selfimprovement Branch
```
Concurrent Users:   High (FastAPI + async)
Deployment:         Multi-server capable
Horizontal Scale:   Easy (stateless API)
API Integration:    Full REST API
Automation:         Training loop
Monitoring:         Comprehensive
```

---

## Development Workflow

### Main Branch
```
Developer → Edit Code → Run Streamlit → Test Manually
                                          ↓
                                    Fix Issues
                                          ↓
                                      Commit
```

### Selfimprovement Branch
```
Developer → Edit Code → Run Tests → Start API Server
                            ↓              ↓
                      Fix Issues    Test Endpoints
                            ↓              ↓
                       Run Training    Monitor Logs
                            ↓              ↓
                   Review Generations  Validate
                            ↓              ↓
                       Commit & Deploy
```

---

## Deployment Comparison

### Main Branch Deployment
```bash
# Simple deployment
git clone repo
cd repo
pip install -r requirements.txt
streamlit run app.py
```

### Selfimprovement Branch Deployment
```bash
# Complex deployment
git clone repo
cd repo
pip install -r requirements.txt

# Set up environment
export GOOGLE_API_KEY=xxx
export OPENAI_API_KEY=xxx
export GROQ_API_KEY=xxx

# Initialize data
mkdir -p data logs

# Start services
python -m src.main serve --host 0.0.0.0 --port 8000 &

# Optional: Start training
python -m src.main train \
    --prompts data/prompts.json \
    --dataset data/prompt_train_cases.json \
    --epochs 10

# Optional: Streamlit UI
streamlit run src/streamlit_app.py --server.port 8501 &
```

---

## Summary

| Dimension | Main | Selfimprovement |
|-----------|------|-----------------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐⭐ Complex |
| **Features** | ⭐⭐ Basic | ⭐⭐⭐⭐⭐ Advanced |
| **Setup Time** | ⭐⭐⭐⭐⭐ 5 min | ⭐⭐ 30+ min |
| **Learning Curve** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐ Steep |
| **Scalability** | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ High |
| **Maintenance** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐ Complex |
| **API Access** | ⭐ None | ⭐⭐⭐⭐⭐ Full |
| **Automation** | ⭐ None | ⭐⭐⭐⭐⭐ Full |
| **Production Ready** | ⭐⭐ Prototype | ⭐⭐⭐⭐ Near Ready |

---

**Visualization Guide:**
- ✅ = Feature present
- ❌ = Feature absent
- ⚠️ = Consideration needed
- ⭐ = Rating (1-5 stars)
