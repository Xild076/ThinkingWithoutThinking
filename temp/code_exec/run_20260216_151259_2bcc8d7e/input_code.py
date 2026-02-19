import sympy as sp
r = sp.symbols('r')
poly = r**2 - 6*r + 9
roots = sp.solve(poly, r)
result = roots
print(result)