import pandas as pd
import numpy as np
data = pd.Series([10,12,12,13,12,14,13,15,14,13,12,11,10,9,8,7,6,5,4,3])
window = 5
rolling_mean = data.rolling(window=window, min_periods=1).mean()
rolling_std = data.rolling(window=window, min_periods=1).std(ddof=0)
z_scores = (data - rolling_mean) / rolling_std
threshold = 3.0
flags = np.abs(z_scores) > threshold
result = flags.sum()
print(result)