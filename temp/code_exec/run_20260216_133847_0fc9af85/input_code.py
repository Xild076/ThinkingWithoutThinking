import sympy as sp
x = sp.Symbol('x')
coeffs = [1, 0, 2, 0, 3]
horner = coeffs[0]
for c in coeffs[1:]:
    horner = horner * x + c
 degree = len(coeffs) - 1
 power = 2
 coeff_index = degree - power
 result = coeffs[coeff_index]
print(result)