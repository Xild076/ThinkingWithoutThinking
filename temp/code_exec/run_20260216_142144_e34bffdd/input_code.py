import pandas as pd
import os

# Create sample data if missing
if not os.path.isfile('data.csv'):
    pd.DataFrame({'entity': ['A','A','B','B'], 'value': [1,2,3,4]}).to_csv('data.csv', index=False)

stats = {}
for chunk in pd.read_csv('data.csv', chunksize=1000):
    for _, row in chunk.iterrows():
        key = row['entity']
        x = row['value']
        if key not in stats:
            stats[key] = {'n':0, 'mean':0.0, 'M2':0.0}
        d = stats[key]
        d['n'] += 1
        delta = x - d['mean']
        d['mean'] += delta / d['n']
        delta2 = x - d['mean']
        d['M2'] += delta * delta2

# Compute variance for entity 'A' (population)
n_A = stats.get('A', {}).get('n', 0)
M2_A = stats.get('A', {}).get('M2', 0.0)
result = M2_A / n_A if n_A > 0 else 0.0
print(result)