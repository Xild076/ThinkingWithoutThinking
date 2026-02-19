import sympy as sp
x = sp.symbols('x')
expr = (1 - x)**(-3) * (1 + x)**(-2)
series = sp.series(expr, x, 0, 11).removeO()
coeff_series = sp.expand(series).coeff(x, 10)
pf = sp.apart(expr, x)
coeff_pf = sp.series(pf, x, 0, 11).removeO().coeff(x, 10)
result = (coeff_series, coeff_pf)
print(result)