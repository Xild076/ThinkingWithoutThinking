import sympy as sp
import numpy as np

n = 50
x = sp.Symbol('x')
poly_exact = sp.expand((1 - x)**n)
coeffs_exact = [sp.expand(poly_exact).coeff(x, k) for k in range(n+1)]
coeffs_float = np.array([int(c) for c in coeffs_exact], dtype=np.float64)
result = np.polyval(coeffs_float, 1.0)
overflow_val = float((10**6)**60)
print(result)