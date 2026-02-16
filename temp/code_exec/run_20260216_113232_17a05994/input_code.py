import sympy as sp
n = sp.symbols('n', integer=True, nonnegative=True)
I = sp.summation((-1)**n/(n+1)**2, (n, 0, sp.oo))
result = sp.simplify(I)
print(result)