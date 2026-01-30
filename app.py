import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.pipeline import Pipeline
from src.pipeline_blocks import get_prompts
from src.cost_tracker import cost_tracker


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)


pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse("ui/index.html")


@app.post("/stream")
async def stream_pipeline(request: dict):
    prompt = request.get("prompt", "").strip()
    thinking_level = request.get("thinking_level", "medium_synth")
    
    if not prompt:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Prompt cannot be empty'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    def event_stream():
        try:
            for event in pipeline.stream(prompt, thinking_level):
                yield f"data: {json.dumps(event, cls=JSONEncoder)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reload-prompts")
async def reload_prompts():
    """Force reload prompts from disk for hot-reloading during development."""
    try:
        prompts = get_prompts(force_reload=True)
        return {"status": "ok", "prompt_count": len(prompts), "prompt_ids": list(prompts.keys())}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/cost-stats")
async def get_cost_stats():
    """Get current session and historical cost statistics."""
    try:
        session_stats = cost_tracker.get_session_stats()
        historical_stats = cost_tracker.get_historical_stats(days=7)
        return {
            "status": "ok",
            "session": session_stats,
            "historical": historical_stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}
