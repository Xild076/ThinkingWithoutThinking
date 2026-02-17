import pandas as pd
import numpy as np
conditions = df['condition'].unique()
tof_values = []
for cond in conditions:
    sub = df[df['condition'] == cond]
    sub = sub.sort_values('time')
    dt = np.gradient(sub['time'].values)
    dX = np.gradient(sub['conversion'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.where(dt != 0, dX / dt, 0)
    tof_values.append(np.mean(slopes))
result = np.mean(tof_values)
print(result)