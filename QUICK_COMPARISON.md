# Quick Branch Comparison Reference

## At a Glance

| Aspect | Main Branch | Selfimprovement Branch |
|--------|-------------|------------------------|
| **Commits ahead** | - | 1 commit |
| **Files changed** | - | 651 files |
| **Lines added** | - | +73,202 |
| **Lines removed** | - | -5,123 |
| **Primary Language** | Python | Python |
| **Main Framework** | Streamlit | FastAPI + Streamlit |
| **Status** | Stable experimental | Experimental rewrite |

## Quick Decision Matrix

### Choose MAIN if you need:
- ✅ Simple Streamlit app
- ✅ Quick prototyping
- ✅ Minimal setup
- ✅ Stable codebase
- ✅ Fewer dependencies

### Choose SELFIMPROVEMENT if you need:
- ✅ REST API endpoints
- ✅ Automated prompt training
- ✅ Production-grade architecture
- ✅ CLI tooling
- ✅ Advanced monitoring
- ✅ Prompt versioning
- ✅ Fault tolerance

## Key Files Comparison

### Entry Points

**Main:**
```bash
streamlit run app.py
```

**Selfimprovement:**
```bash
# API Server
python -m src.main serve --port 8000

# CLI
python -m src.main run --prompt "..." --thinking med-synth

# Training
python -m src.main train --prompts data/prompts.json --dataset data/prompt_train_cases.json
```

### File Structure

**Main (8 core files):**
```
app.py
src/thinking_pipeline.py
src/pipeline_blocks.py
src/pdf_utils.py
src/utility.py
requirements.txt
README.md
.gitignore
```

**Selfimprovement (30+ core files):**
```
src/app.py                    # FastAPI server
src/main.py                   # CLI entrypoint
src/pipeline.py               # Pipeline orchestration
src/pipeline_blocks.py        # Modular blocks
src/prompt_training.py        # Training loop
src/streamlit_app.py          # Streamlit UI
src/rate_limit_monitoring.py # Monitoring
src/ui/index.html             # Pipeline dashboard
src/ui/training.html          # Training dashboard
src/tools/                    # Tool modules
data/                         # Training data
logs/                         # Application logs
PLAN.MD
writeup.md
```

## New Features in Selfimprovement

| Feature | Description | Files |
|---------|-------------|-------|
| **REST API** | Full REST API with FastAPI | `src/app.py` |
| **CLI Tool** | Command-line interface | `src/main.py` |
| **Prompt Training** | Automated A/B testing & optimization | `src/prompt_training.py` |
| **Web Dashboards** | Browser-based monitoring | `src/ui/*.html` |
| **Tool Modules** | Organized tool implementations | `src/tools/*.py` |
| **Rate Limiting** | API rate limit monitoring | `src/rate_limit_monitoring.py` |
| **Training Data** | Structured training datasets | `data/*.json` |
| **Logging System** | Structured application logs | `logs/*.log` |
| **Telemetry** | Ingestion architecture | Throughout |
| **Fault Tolerance** | Enhanced error handling | Throughout |

## Dependencies Added

```python
# Selfimprovement adds:
fastapi        # Web framework
openai         # OpenAI API client
groq           # Groq API client  
uvicorn        # ASGI server
```

## Migration Effort Estimate

| Task | Estimated Effort | Complexity |
|------|-----------------|------------|
| **Code Migration** | 2-3 days | High |
| **Data Migration** | 1 day | Medium |
| **Testing** | 2-3 days | High |
| **Documentation** | 1-2 days | Medium |
| **Training Setup** | 1-2 days | Medium |
| **Deployment** | 1 day | Medium |
| **Total** | **8-12 days** | **High** |

## Breaking Changes

⚠️ **All of these break backwards compatibility:**

1. ❌ Different entry point (`src/main.py` vs `app.py`)
2. ❌ Different architecture (FastAPI vs Streamlit-only)
3. ❌ Different configuration format
4. ❌ Different data structures
5. ❌ Different API/interface
6. ❌ Different deployment model
7. ❌ Different monitoring approach

## Risks

### Main Branch
- Limited scalability
- No API access
- Manual prompt optimization
- Basic error handling

### Selfimprovement Branch
- ⚠️ Untested in production
- ⚠️ Complex architecture
- ⚠️ Single massive commit
- ⚠️ Temp files committed to Git
- ⚠️ `__pycache__` committed
- ⚠️ Unpinned dependencies
- ⚠️ Migration complexity

## Testing

### Main Branch
```bash
pytest -q
```

### Selfimprovement Branch
```bash
pytest -q
# Plus need to test:
# - API endpoints
# - Training loop
# - CLI commands
# - Tool modules
```

## Repository Statistics

### Main Branch
- **Total files tracked:** 27
- **Python files:** 5
- **Test files:** 2
- **Config files:** 2

### Selfimprovement Branch  
- **Total files tracked:** 651
- **Python files:** 11 (core, excluding temp)
- **Temp files:** 600+ (should not be committed)
- **Data files:** 7
- **UI files:** 2
- **Log files:** 2

## Next Steps

### If Choosing Main
1. Continue iterative development
2. Add features incrementally
3. Maintain Streamlit focus

### If Choosing Selfimprovement
1. Clean up repository (remove temp/, __pycache__)
2. Pin dependency versions
3. Add comprehensive tests
4. Write migration documentation
5. Create deployment guide
6. Plan incremental rollout
7. Set up staging environment
8. Perform thorough testing

## Timeline Comparison

### Main Branch Development
- Multiple incremental commits
- Clear feature history
- Easy to review changes
- Can bisect issues

### Selfimprovement Branch Development
- Single massive commit
- Lost development history
- Difficult to review
- Cannot bisect features

## Recommendation Summary

**For most teams:** Start with MAIN and add features incrementally

**For advanced needs:** Consider SELFIMPROVEMENT but:
1. Clean up the repository first
2. Break down into incremental commits
3. Add comprehensive tests
4. Create proper documentation
5. Plan careful migration

---

**Last Updated:** February 17, 2026  
**Compared:** selfimprovement (ad2b2d3) vs main (21f8ee0)
