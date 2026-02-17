from collections import defaultdict
import sympy as sp

chunks = [
    ('A', 1),
    ('B', 2),
    ('A', 3),
    ('B', 4),
    ('A', 5),
]

counts = defaultdict(int)
sums = defaultdict(sp.Integer(0))
sum_sqs = defaultdict(sp.Integer(0))

for entity, val in chunks:
    x = sp.Integer(val)
    counts[entity] += 1
    sums[entity] += x
    sum_sqs[entity] += x**2

entity = 'A'
n = counts[entity]
mean = sums[entity] / n
var = (sum_sqs[entity] - sums[entity]**2 / n) / n
result = var
print(result)
