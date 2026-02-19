import numpy as np
import pandas as pd

data = pd.Series([10, 12, 12, 13, 12, 14, 13, 15, 100])
limit = 2
z_scores = (data - data.mean()) / data.std()
flagged = np.abs(z_scores) > limit
result = flagged.sum()
print(result)