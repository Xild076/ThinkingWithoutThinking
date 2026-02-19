import pandas as pd
import numpy as np
threshold = 3
window = 5
data = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame({'value': data})
rolling_mean = df['value'].rolling(window).mean()
rolling_std = df['value'].rolling(window).std()
z_score = (df['value'] - rolling_mean) / rolling_std
anomalies = df[abs(z_score) > threshold].dropna()
result = len(anomalies)
print(result)