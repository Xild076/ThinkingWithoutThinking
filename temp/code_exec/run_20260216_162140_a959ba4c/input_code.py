import sympy as sp
result = sp.N(sp.summation(1/sp.Symbol('n')**2, (sp.Symbol('n'), 1, 10)))
print(result)