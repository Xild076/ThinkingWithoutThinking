import sympy as sp
x = sp.Symbol('x')
expr = (1 + x)**-2
series = sp.series(expr, x, 0, 11).removeO()
coeffs = [series.coeff(x, i) for i in range(11)]
result = coeffs
print(result)