import numpy as np
import sympy as sp
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
z_scores = []
for i, x in enumerate(data):
    if i == 0:
        mean = sp.Rational(x)
        std = 0
    else:
        subset = data[:i+1]
        subset_sym = [sp.Rational(val) for val in subset]
        mean = sp.mean(subset_sym)
        variance = sp.mean([ (xi - mean)**2 for xi in subset_sym ])
        std = sp.sqrt(variance)
    if std != 0:
        z = (sp.Rational(x) - mean) / std
    else:
        z = 0
    z_scores.append(sp.N(z))
result = z_scores[-1]
print(result)