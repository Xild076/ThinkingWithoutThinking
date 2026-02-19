import pandas as pd
import numpy as np

try:
    df = pd.read_csv('streaming.csv')
except FileNotFoundError:
    df = pd.DataFrame({
        'value': [10, 12, 12, 14, 13, 15, 9],
        'ref_anomaly': [False, False, True, False, False, False, False]
    })

z = (df['value'] - df['value'].mean()) / df['value'].std(ddof=0)
anomaly = np.abs(z) > 3
result = (anomaly != df['ref_anomaly']).sum()
print(result)