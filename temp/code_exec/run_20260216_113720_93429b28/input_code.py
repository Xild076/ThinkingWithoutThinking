import sympy as sp
x = sp.symbols('x')
f = 1/((1 - x)**2 * (1 + x))
coeff_series = sp.series(f, x, 0, 11).removeO().coeff(x, 10)
pf = sp.apart(f, x)
coeff_pf = sum(term.series(x, 0, 11).removeO().coeff(x, 10) for term in pf.as_ordered_terms())
result = coeff_series == coeff_pf
print(result)