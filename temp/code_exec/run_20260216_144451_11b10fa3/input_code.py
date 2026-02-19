import pandas as pd
import numpy as np
if 'df' not in globals():
    dates = pd.date_range('2023-01-01', periods=6, freq='D')
    entities = ['A', 'B']
    index = pd.MultiIndex.from_product([entities, dates], names=['entity','time'])
    df = pd.DataFrame({'col1': np.arange(1,13), 'col2': np.arange(13,25)}, index=index)
window = 3
rolling_mean = df.rolling(window=window, min_periods=1).mean()
rolling_std = df.rolling(window=window, min_periods=1).std()
rolling_std = rolling_std.replace(0, np.nan)
z_score = (df - rolling_mean) / rolling_std
result = z_score
print(result)