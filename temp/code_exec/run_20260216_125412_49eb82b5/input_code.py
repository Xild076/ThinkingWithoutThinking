import sympy as sp
x = sp.symbols('x')
expr = sp.exp(-x**2)
res = sp.integrate(expr, (x, 0, 1))
result = res.evalf()
print(result)