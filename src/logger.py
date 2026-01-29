import logging
import sys
from typing import Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ExecutionTrace:
    def __init__(self):
        self.events = []
        self.errors = []
        self.start_time = datetime.now()

    def log(self, stage: str, message: str, data: dict[str, Any] = None):
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "data": data or {}
        }
        self.events.append(event)
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
        logger = logging.getLogger(f"pipeline.{stage}")
        logger.error(f"ERROR: {error} | {context or ''}")

    def get_summary(self):
        return {
            "total_events": len(self.events),
            "total_errors": len(self.errors),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "events": self.events,
            "errors": self.errors
        }

    def get_full_trace(self):
        excluded_stages = {"tool_invoke"}
        return [event for event in self.events if event.get("stage") not in excluded_stages]
