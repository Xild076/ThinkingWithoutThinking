import sympy as sp
n = 5
k = sp.symbols('k', integer=True, positive=True)
result = sp.summation(k*2**k, (k, 1, n))
print(result)