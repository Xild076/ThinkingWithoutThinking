import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)**3 * (1+x)**2)
coeff = sp.series(expr, x, 0, 6).removeO().coeff(x, 5)
result = coeff
print(result)