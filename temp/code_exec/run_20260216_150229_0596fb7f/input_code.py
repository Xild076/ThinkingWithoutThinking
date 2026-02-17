import sympy as sp
k = sp.symbols('k', integer=True, positive=True)
expr = sp.summation(k * 2**k, (k, 1, 20))
result = sp.simplify(expr)
print(result)