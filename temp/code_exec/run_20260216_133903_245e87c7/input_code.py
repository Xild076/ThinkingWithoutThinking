import sympy as sp
x = sp.Symbol('x')
coeffs = [1, 0, 2, 0, 3]
poly = sp.Integer(coeffs[0])
for c in coeffs[1:]:
    poly = poly * x + sp.Integer(c)
power = 2
coeff = sp.Poly(poly, x).coeff_monomial(x**power)
result = coeff
print(result)