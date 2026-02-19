import sympy as sp
n = sp.symbols('n', integer=True, nonnegative=True)
a_n = sp.binomial(2*n, n) / (n+1)
result = [sp.simplify(a_n.subs(n,i)) for i in range(11)]
print(result)