import pandas as pd
import numpy as np
from collections import deque
from io import StringIO

csv_data = '''
entity,date,value
A,2023-01-01,1.0
A,2023-01-02,2.0
A,2023-01-03,3.0
A,2023-01-04,4.0
A,2023-01-05,5.0
A,2023-01-06,6.0
A,2023-01-07,7.0
A,2023-01-08,8.0
A,2023-01-09,9.0
A,2023-01-10,10.0
A,2023-01-11,11.0
A,2023-01-12,1000.0
B,2023-01-01,100.0
B,2023-01-02,101.0
B,2023-01-03,102.0
B,2023-01-04,103.0
B,2023-01-05,104.0
B,2023-01-06,105.0
B,2023-01-07,106.0
B,2023-01-08,107.0
B,2023-01-09,108.0
B,2023-01-10,109.0
B,2023-01-11,110.0
'''

chunksize = 2
dtype = {'value':'float64'}
parse_dates = ['date']
window_size = 10
threshold = 3
anomalies = 0
windows = {}
for chunk in pd.read_csv(StringIO(csv_data), chunksize=chunksize, dtype=dtype, parse_dates=parse_dates):
    for _, row in chunk.iterrows():
        entity = row['entity']
        value = row['value']
        if entity not in windows:
            windows[entity] = deque(maxlen=window_size)
        dq = windows[entity]
        dq.append(value)
        if len(dq) == window_size:
            mean = np.mean(dq)
            std = np.std(dq)
            if std != 0:
                z = (value - mean) / std
                if abs(z) > threshold:
                    anomalies += 1
result = anomalies
print(result)