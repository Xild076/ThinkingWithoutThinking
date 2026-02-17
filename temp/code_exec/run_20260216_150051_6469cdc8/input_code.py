import sympy as sp
k = sp.symbols('k', integer=True, positive=True)
result = sp.summation(k*2**k, (k, 1, 20))
print(result)