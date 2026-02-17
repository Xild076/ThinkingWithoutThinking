import sympy as sp
k = sp.symbols('k', integer=True, positive=True)
n = sp.symbols('n', integer=True, positive=True)
T = sp.summation(1/(k*(k+1)), (k, 1, n))
T_simplified = sp.simplify(T)
result = T_simplified.subs(n, 1000)
print(result)