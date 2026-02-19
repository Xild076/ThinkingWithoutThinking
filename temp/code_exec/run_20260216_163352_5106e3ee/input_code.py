import pandas as pd
import numpy as np
import tempfile
import os
import time
csv_data = '''a,b
c1,10
c2,20
c3,30
c4,100
c5,150
c6,10
c7,20
c8,30
c9,100
c10,150'''
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
    f.write(csv_data)
    temp_path = f.name
Desired_rate = 0.8
window_size = 3
threshold = 2.0
total_anomalies = 2
while True:
    detection_count = 0
    values = []
    for chunk in pd.read_csv(temp_path, chunksize=2):
        values.extend(chunk['b'].tolist())
    for i in range(len(values)):
        if i >= window_size:
            window = values[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                z = 0
            else:
                z = (values[i] - mean) / std
            if abs(z) > threshold:
                detection_count += 1
    achieved_rate = detection_count / total_anomalies
    if achieved_rate >= Desired_rate:
        break
    window_size += 1
result = achieved_rate
print(result)
os.remove(temp_path)
