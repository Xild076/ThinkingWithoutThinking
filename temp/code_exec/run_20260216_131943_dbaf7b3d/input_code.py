import sympy as sp
import numpy as np
n = 50
coeffs = [sp.binomial(n, k) for k in range(n+1)]
coeffs_np = np.array([float(c) for c in coeffs])
V = np.vander(np.arange(n+1), N=n+1, increasing=True)
cond = np.linalg.cond(V)
result = cond
print(result)