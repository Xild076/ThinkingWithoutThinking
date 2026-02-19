import sympy as sp
x = sp.symbols('x')
expr = x**2 * sp.log(1+x)
I = sp.integrate(expr, (x, 0, 1))
result = sp.simplify(I)
print(result)