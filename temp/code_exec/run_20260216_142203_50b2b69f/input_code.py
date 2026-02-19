import sympy as sp
from collections import defaultdict
threshold = 2
data = [{'entity': 'A', 'value': 10}, {'entity': 'A', 'value': 12}, {'entity': 'A', 'value': 14}, {'entity': 'B', 'value': 5}, {'entity': 'B', 'value': 7}, {'entity': 'B', 'value': 9}]
stats = defaultdict(lambda: {'count': 0, 'mean': 0, 'M2': 0.0})
exceed = 0
for rec in data:
    x = rec['value']
    entity = rec['entity']
    count = stats[entity]['count']
    if count == 0:
        new_mean = x
        new_M2 = 0.0
    else:
        old_mean = stats[entity]['mean']
        new_mean = old_mean + (x - old_mean) / (count + 1)
        new_M2 = stats[entity]['M2'] + (x - old_mean) * (x - new_mean)
    stats[entity]['count'] = count + 1
    stats[entity]['mean'] = new_mean
    stats[entity]['M2'] = new_M2
    var = sp.N(new_M2 / (count + 1))
    std = sp.sqrt(var)
    z = (x - new_mean) / std
    if sp.Abs(z) > threshold:
        exceed += 1
result = exceed
print(result)