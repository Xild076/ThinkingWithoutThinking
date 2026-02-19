import sympy as sp
n = sp.symbols('n', integer=True, nonnegative=True)
sum_term = sp.summation((-1)**n/(n+2), (n, 0, sp.oo))
integral_val = sp.log(2) - sum_term
result = sp.simplify(integral_val)
print(result)