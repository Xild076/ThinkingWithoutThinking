def execute_with_retry(func, max_retries=3, idempotency_key=None):
    executed_keys = set()
    attempts = 0
    while attempts < max_retries:
        if idempotency_key and idempotency_key in executed_keys:
            raise Exception('Duplicate execution')
        try:
            result = func()
            if idempotency_key:
                executed_keys.add(idempotency_key)
            return result
        except Exception:
            attempts += 1
            if attempts == max_retries:
                raise
    return None

def core_engine():
    def inner():
        return 42
    return execute_with_retry(inner)

result = core_engine()
print(result)