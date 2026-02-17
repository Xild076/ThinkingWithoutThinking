import pandas as pd
import numpy as np
if 'df' not in globals():
    dates = pd.date_range('2023-01-01', periods=6, freq='D')
    entities = ['A', 'B']
    data = {('A','col1'): [1,2,3,4,5,6], ('A','col2'): [2,4,6,8,10,12], ('B','col1'): [5,7,9,11,13,15], ('B','col2'): [10,20,30,40,50,60]}
    index = pd.MultiIndex.from_product([entities, dates], names=['entity','time'])
    df = pd.DataFrame(data, index=index)
window = 3
rolling_mean = df.rolling(window=window, min_periods=1).mean()
rolling_std = df.rolling(window=window, min_periods=1).std()
rolling_std = rolling_std.replace(0, np.nan)
z_score = (df - rolling_mean) / rolling_std
result = z_score
print(result)