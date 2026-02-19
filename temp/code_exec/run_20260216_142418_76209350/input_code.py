import pandas as pd
import numpy as np
import scipy.stats as stats
import gc

n_entities = 5
rows_per_entity = 200000
anomaly_frac = 0.01
z_thresh = 3
chunk_size = 50000
expected_anomalies = set()
frames = []
for e in range(n_entities):
    values = np.random.normal(size=rows_per_entity)
    n_anom = int(rows_per_entity * anomaly_frac)
    anom_idx = np.random.choice(rows_per_entity, n_anom, replace=False)
    values[anom_idx] += 10
    df = pd.DataFrame({'entity': e, 'value': values})
    start = e * rows_per_entity
    for pos in anom_idx:
        expected_anomalies.add(start + pos)
    frames.append(df)
df_all = pd.concat(frames, ignore_index=True)
csv_path = 'synthetic.csv'
df_all.to_csv(csv_path, index=False)
flagged_anomalies = set()
row_counter = 0
for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    roll = chunk['value'].rolling(window=100, min_periods=1)
    mean = roll.mean()
    std = roll.std(ddof=0)
    z = (chunk['value'] - mean) / std
    is_anom = z.abs() > z_thresh
    for local_idx in chunk[is_anom].index:
        global_idx = row_counter + local_idx
        flagged_anomalies.add(global_idx)
    row_counter += len(chunk)
    del chunk
    gc.collect()
mismatch = flagged_anomalies.symmetric_difference(expected_anomalies)
result = len(mismatch)
print(result)