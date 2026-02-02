import atexit
import json
import logging
import os
import sys
from typing import Any
from datetime import datetime
from uuid import uuid4

_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_TRACE_LOG_DIR = os.getenv("TRACE_LOG_DIR", "data/trace_logs")
_PIPELINE_LOG_PATH = os.getenv("PIPELINE_LOG_PATH", "data/pipeline.log")

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
else:
    root_logger.setLevel(logging.INFO)

os.makedirs(os.path.dirname(_PIPELINE_LOG_PATH), exist_ok=True)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(_PIPELINE_LOG_PATH)
           for h in root_logger.handlers):
    file_handler = logging.FileHandler(_PIPELINE_LOG_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(file_handler)

class ExecutionTrace:
    def __init__(self):
        self.events = []
        self.errors = []
        self.start_time = datetime.now()
        os.makedirs(_TRACE_LOG_DIR, exist_ok=True)
        trace_id = f"{self.start_time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid4().hex[:8]}"
        self.trace_path = os.path.join(_TRACE_LOG_DIR, f"trace_{trace_id}.jsonl")
        self._trace_fp = open(self.trace_path, "a", encoding="utf-8")
        atexit.register(self._close)

    def _close(self):
        if getattr(self, "_trace_fp", None):
            try:
                self._trace_fp.close()
            except Exception:
                pass
            self._trace_fp = None

    def _write_event(self, event: dict[str, Any]):
        if not getattr(self, "_trace_fp", None):
            return
        try:
            self._trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._trace_fp.flush()
        except Exception:
            pass

    def log(self, stage: str, message: str, data: dict[str, Any] = None):
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "data": data or {}
        }
        self.events.append(event)
        self._write_event(event)
        logger = logging.getLogger(f"pipeline.{stage}")
        logger.info(f"{message} | {data or ''}")

    def log_error(self, stage: str, error: str, context: dict[str, Any] = None):
        error_event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error": error,
            "context": context or {}
        }
        self.errors.append(error_event)
        self._write_event({"timestamp": error_event["timestamp"], "stage": stage, "message": error, "data": context or {}, "error": True})
        logger = logging.getLogger(f"pipeline.{stage}")
        logger.error(f"ERROR: {error} | {context or ''}")

    def log_trace(self, stage: str, data: dict[str, Any] = None, message: str = ""):
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "data": data or {},
            "trace_only": True
        }
        self.events.append(event)
        self._write_event(event)

    def get_summary(self):
        return {
            "total_events": len(self.events),
            "total_errors": len(self.errors),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "events": self.events,
            "errors": self.errors,
            "trace_path": self.trace_path
        }

    def get_full_trace(self):
        full = list(self.events)
        for err in self.errors:
            full.append({
                "timestamp": err.get("timestamp"),
                "stage": err.get("stage"),
                "message": err.get("error"),
                "data": err.get("context", {}),
                "error": True
            })
        return full
