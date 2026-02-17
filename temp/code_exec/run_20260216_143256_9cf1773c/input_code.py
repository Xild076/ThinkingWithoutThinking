import sympy as sp
max_retries = 3
events = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
idempotency_keys = {'Alpha': 'kA', 'Beta': 'kB', 'Gamma': 'kA', 'Delta': 'kD', 'Epsilon': 'kE'}
processed_keys = set()
successful = 0
poison = 0
for event in events:
    key = idempotency_keys[event]
    if key in processed_keys:
        continue
    processed_keys.add(key)
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        if len(event) % 2 == 0:
            successful += 1
            break
        if attempts == max_retries:
            poison += 1
result = sp.Rational(successful, 2) + sp.Rational(1, 3)
result = sp.simplify(result)
print(result)