import numpy as np
import pandas as pd
from scipy import stats

loads = np.linspace(0.1, 0.9, 9)
latencies = 10 + 50 * loads
error_rates = np.clip(0.01 + 0.05 * loads, 0, 0.2)
cutover_times = 2 + 1 * loads

df = pd.DataFrame({
    'load': loads,
    'latency_ms': latencies,
    'error_rate': error_rates,
    'cutover_s': cutover_times
})

avg_cutover = df['cutover_s'].mean()
result = avg_cutover
print(result)