import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('streaming.csv')
z = (df['value'] - df['value'].mean()) / df['value'].std(ddof=0)
anomaly = np.abs(z) > 3
ref_anomaly = df['ref_anomaly']
result = (anomaly != ref_anomaly).sum()
print(result)