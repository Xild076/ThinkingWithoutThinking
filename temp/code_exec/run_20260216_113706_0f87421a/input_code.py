import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)*(1-2*x)*(1-3*x))
pf = sp.apart(expr, x)
series = sp.series(pf, x, 0, 11).removeO()
coeff = sp.expand(series).coeff(x, 10)
result = coeff
print(result)