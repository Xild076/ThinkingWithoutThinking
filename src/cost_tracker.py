import csv
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

COST_LOG_PATH = "data/cost_tracking.csv"

MODEL_COSTS = {
    "gemma": {"input": 0.0, "output": 0.0},
    "nemotron": {"input": 0.0, "output": 0.0},
}

@dataclass
class APICall:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    latency_ms: float
    success: bool
    error: str = ""


class CostTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.calls: list[APICall] = []
        self.session_start = datetime.now()
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        os.makedirs(os.path.dirname(COST_LOG_PATH), exist_ok=True)
        if not os.path.exists(COST_LOG_PATH):
            with open(COST_LOG_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "model", "input_tokens", "output_tokens",
                    "input_cost", "output_cost", "total_cost", "latency_ms",
                    "success", "error"
                ])

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: str = ""
    ):
        costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        self.calls.append(call)
        self._write_to_csv(call)

    def _write_to_csv(self, call: APICall):
        with open(COST_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                call.timestamp, call.model, call.input_tokens, call.output_tokens,
                f"{call.input_cost:.6f}", f"{call.output_cost:.6f}", f"{call.total_cost:.6f}",
                f"{call.latency_ms:.2f}", call.success, call.error
            ])

    def get_session_stats(self) -> dict:
        total_calls = len(self.calls)
        successful_calls = sum(1 for c in self.calls if c.success)
        total_input_tokens = sum(c.input_tokens for c in self.calls)
        total_output_tokens = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.total_cost for c in self.calls)
        avg_latency = sum(c.latency_ms for c in self.calls) / max(total_calls, 1)

        by_model: dict[str, dict] = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost": 0.0, "errors": 0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["input_tokens"] += call.input_tokens
            by_model[call.model]["output_tokens"] += call.output_tokens
            by_model[call.model]["cost"] += call.total_cost
            if not call.success:
                by_model[call.model]["errors"] += 1

        return {
            "session_start": self.session_start.isoformat(),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "by_model": by_model
        }

    def get_historical_stats(self, days: int = 7) -> dict:
        stats = {
            "total_calls": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "by_day": {}
        }
        
        if not os.path.exists(COST_LOG_PATH):
            return stats

        cutoff = datetime.now().timestamp() - (days * 86400)
        
        with open(COST_LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts.timestamp() < cutoff:
                        continue
                    
                    day = ts.strftime("%Y-%m-%d")
                    if day not in stats["by_day"]:
                        stats["by_day"][day] = {"calls": 0, "cost": 0.0, "tokens": 0}
                    
                    stats["total_calls"] += 1
                    stats["total_cost"] += float(row["total_cost"])
                    tokens = int(row["input_tokens"]) + int(row["output_tokens"])
                    stats["total_tokens"] += tokens
                    
                    stats["by_day"][day]["calls"] += 1
                    stats["by_day"][day]["cost"] += float(row["total_cost"])
                    stats["by_day"][day]["tokens"] += tokens
                except (ValueError, KeyError):
                    continue

        stats["total_cost"] = round(stats["total_cost"], 6)
        return stats

    def reset_session(self):
        self.calls = []
        self.session_start = datetime.now()


cost_tracker = CostTracker()
