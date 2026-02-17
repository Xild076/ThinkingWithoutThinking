import sympy as sp
import numpy as np
n = 50
coeffs = [sp.binomial(n, k) * (-1)**k for k in range(n+1)]
result_exact = sum(coeffs)
result = float(result_exact)
print(result)