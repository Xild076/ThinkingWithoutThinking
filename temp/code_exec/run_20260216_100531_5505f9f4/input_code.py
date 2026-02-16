import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)*(1-2*x))
pf = sp.apart(expr, x)
terms = list(pf.as_ordered_terms())
coeffs = []
for term in terms:
    ser = term.series(x, 0, 11).removeO()
    coeffs.append(ser.coeff(x, 10))
result = sum(coeffs)
print(result)