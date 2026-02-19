import pandas as pd
import numpy as np
import io

# generate synthetic CSV data
np.random.seed(0)
df = pd.DataFrame({'value': np.random.randn(20000)})
# inject anomalies
anomaly_idx = np.random.choice(df.index, 200, replace=False)
df.loc[anomaly_idx, 'value'] += 5
csv_data = df.to_csv(index=False)

# streaming parameters
chunk_size = 2000
target_rate = 0.8
best_det_rate = 0
best_w = None
best_t = None

# iterate over window sizes and thresholds
for w in range(100, 501, 100):
    for t in np.arange(1, 4.5, 0.5):
        detections = 0
        total_anom = 200
        for chunk in pd.read_csv(io.StringIO(csv_data), chunksize=chunk_size):
            roll = chunk['value'].rolling(w).mean()
            std = chunk['value'].rolling(w).std()
            is_anom = np.abs(chunk['value'] - roll) > t * std
            detections += is_anom.sum()
        det_rate = detections / total_anom
        if det_rate >= target_rate and w * chunk_size < 10**6:
            if det_rate > best_det_rate:
                best_det_rate = det_rate
                best_w = w
                best_t = t

# final result
result = best_det_rate
print(result)