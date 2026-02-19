import sympy as sp
x = sp.symbols('x')
binomial_coeff = sp.binomial(10, 10)
expr = 1/(1 - x)**2
partial_frac_coeff = sp.series(expr, x, 0, 11).removeO().coeff(x, 10)
match = binomial_coeff == partial_frac_coeff
result = {"binomial_coeff": binomial_coeff, "partial_frac_coeff": partial_frac_coeff, "match": match}
print(result)