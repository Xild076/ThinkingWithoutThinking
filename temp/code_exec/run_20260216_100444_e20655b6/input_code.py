import sympy as sp
k = sp.symbols('k', integer=True, nonnegative=True)
c1 = sp.binomial(k+2, 2)
c2 = (-1)**k * (k+1)
coeffs1 = [sp.binomial(i+2, 2) for i in range(11)]
coeffs2 = [(-1)**i * (i+1) for i in range(11)]
result = sum(coeffs1) + sum(coeffs2)
print(result)