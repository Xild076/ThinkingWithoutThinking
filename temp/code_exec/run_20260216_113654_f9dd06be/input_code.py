import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)**3 * (1+x)**2)
partial = sp.apart(expr, x)
series_terms = [sp.series(term, x, 0, 11).removeO() for term in partial.as_ordered_terms()]
series_sum = sum(series_terms)
coeff_10 = sp.expand(series_sum).coeff(x, 10)
result = coeff_10
print(result)