import sys
from typing import Callable, Any

# Idempotency tracking
_executed_keys = set()

def execute_with_retry(func: Callable[[], Any], max_retries: int = 3, idempotency_key: str = None) -> Any:
    attempts = 0
    while attempts < max_retries:
        if idempotency_key and idempotency_key in _executed_keys:
            raise Exception('Duplicate execution')
        try:
            result = func()
            if idempotency_key:
                _executed_keys.add(idempotency_key)
            return result
        except Exception as e:
            attempts += 1
            if attempts == max_retries:
                raise
    return None

# Example core function

def core_engine() -> int:
    def inner() -> int:
        return 42
    return execute_with_retry(inner)

result = core_engine()
print(result)
