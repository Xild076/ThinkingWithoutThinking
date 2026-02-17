import sympy as sp
n = sp.symbols('n')
expr = (-3 + sp.Rational(5,3)*n)*3**n + 4*2**n
simplified = sp.simplify(expr)
result = simplified.subs(n, 1)
print(result)