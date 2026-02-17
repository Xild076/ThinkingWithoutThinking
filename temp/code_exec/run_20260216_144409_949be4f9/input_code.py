import pandas as pd
import numpy as np

# Sample data
data = {
    'entity': ['A', 'A', 'A', 'B', 'B', 'B'],
    'col1': [1, 2, 3, 4, 5, 6],
    'col2': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)


def incremental_mean_variance(group):
    means = {}
    vars_ = {}
    for col in group.select_dtypes(include=[np.number]).columns:
        values = group[col].values
        mean = 0.0
        M2 = 0.0
        for i, x in enumerate(values, 1):
            delta = x - mean
            mean += delta / i
            delta2 = x - mean
            M2 += delta * delta2
        if i > 0:
            variance = M2 / (i - 1) if i > 1 else float('nan')
        else:
            variance = float('nan')
        means[col] = mean
        vars_[col] = variance
    return means, vars_

result_dict = {}
for name, g in df.groupby('entity'):
    means, vars_ = incremental_mean_variance(g)
    result_dict[name] = {'mean': means, 'variance': vars_}

result = result_dict
print(result)
