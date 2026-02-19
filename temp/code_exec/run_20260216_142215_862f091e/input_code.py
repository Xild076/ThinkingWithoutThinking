import sympy as sp
from collections import defaultdict

threshold = 2
data = [{'entity': 'A', 'value': 10}, {'entity': 'A', 'value': 12}, {'entity': 'A', 'value': 14}, {'entity': 'B', 'value': 5}, {'entity': 'B', 'value': 7}, {'entity': 'B', 'value': 9}]
stats = defaultdict(lambda: {'count': 0, 'mean': sp.Integer(0), 'M2': sp.Integer(0)})
exceed = 0
for rec in data:
    x = rec['value']
    entity = rec['entity']
    stats_entry = stats[entity]
    count = stats_entry['count']
    if count == 0:
        new_mean = sp.Integer(x)
        new_M2 = sp.Integer(0)
    else:
        old_mean = stats_entry['mean']
        new_mean = old_mean + (sp.Integer(x) - old_mean) / (count + 1)
        new_M2 = stats_entry['M2'] + (sp.Integer(x) - old_mean) * (sp.Integer(x) - new_mean)
    stats_entry['count'] = count + 1
    stats_entry['mean'] = new_mean
    stats_entry['M2'] = new_M2
    var = new_M2 / (count + 1)
    if var == 0:
        z = sp.Integer(0)
    else:
        z = (sp.Integer(x) - new_mean) / sp.sqrt(var)
    if sp.Abs(z) > sp.Integer(threshold):
        exceed += 1
result = exceed
print(result)