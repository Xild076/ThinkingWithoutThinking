import sympy as sp
x = sp.symbols('x')
expr = (1 - x)**(-3) * (1 + x)**(-2)
pf = sp.apart(expr, x)
series_terms = [sp.series(term, x, 0, 11).removeO() for term in pf.as_ordered_terms()]
combined_series = sum(series_terms)
coeff_x10 = sp.expand(combined_series).coeff(x, 10)
result = coeff_x10
print(result)