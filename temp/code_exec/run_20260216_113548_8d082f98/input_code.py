import sympy as sp
x = sp.symbols('x')
expr = (1 - x)**(-3)
result = sp.series(expr, x, 0, 11).removeO()
print(result)