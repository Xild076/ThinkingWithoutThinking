import sympy as sp
n = sp.symbols('n', integer=True, positive=True)
S = sp.summation(sp.Symbol('k')*2**sp.Symbol('k'), (sp.Symbol('k'), 1, n))
closed_form = sp.simplify(S)
result = closed_form.subs(n, 5)
print(result)