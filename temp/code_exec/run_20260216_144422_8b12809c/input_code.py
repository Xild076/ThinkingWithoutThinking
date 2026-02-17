import pandas as pd
import numpy as np
from collections import defaultdict

# Streaming CSV processing
chunksize = 10000
csv_path = 'data.csv'
# Accumulate per-entity per-column sums, sum of squares, and counts
entity_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}))

for chunk in pd.read_csv(csv_path, chunksize=chunksize):
    # Assume the first column is the entity identifier
    entity_col = chunk.iloc[:, 0]
    # All remaining columns are numeric measurements
    numeric_cols = chunk.columns[1:]
    for _, row in chunk.iterrows():
        entity = row.iloc[0]
        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                continue
            entity_stats[entity][col]['sum'] += val
            entity_stats[entity][col]['sum_sq'] += val * val
            entity_stats[entity][col]['count'] += 1

# Compute final mean and variance per entity per column
result_data = []
for entity, cols in entity_stats.items():
    for col, stats in cols.items():
        n = stats['count']
        if n == 0:
            continue
        mean = stats['sum'] / n
        # variance using Welford-like formula: E[X^2] - mean^2
        var = stats['sum_sq'] / n - mean**2
        result_data.append({'entity': entity, 'column': col, 'mean': mean, 'variance': var})

result = pd.DataFrame(result_data)
print(result)
