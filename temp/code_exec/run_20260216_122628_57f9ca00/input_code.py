import sympy as sp
x = sp.symbols('x')
expr = (1 - x)**(-3)*(1 + x)**(-2)
series = sp.series(expr, x, 0, 11).removeO()
coeff = sp.expand(series).coeff(x, 10)
result = coeff
print(result)