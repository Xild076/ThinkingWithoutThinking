import sympy as sp
x = sp.symbols('x')
series1 = sp.series((1 - x)**(-3), x, 0, 11).removeO()
series2 = sp.series((1 + x)**(-2), x, 0, 11).removeO()
coeff1 = sp.expand(series1).coeff(x, 10)
coeff2 = sp.expand(series2).coeff(x, 10)
result = coeff1 + coeff2
print(result)