import sympy as sp
x = sp.symbols('x')
series1 = sp.series(1/(1-x)**3, x, 0, 11).removeO()
series2 = sp.series(1/(1+x)**2, x, 0, 11).removeO()
product = sp.expand(series1 * series2)
coeff_x10 = product.coeff(x, 10)
expr = 1/((1-x)**3 * (1+x)**2)
pf = sp.apart(expr, x)
prose = f"The coefficient of x^10 in (1-x)^(-3)*(1+x)^(-2) is {coeff_x10}. The partial fraction decomposition of the product is {pf}."
result = prose
print(result)