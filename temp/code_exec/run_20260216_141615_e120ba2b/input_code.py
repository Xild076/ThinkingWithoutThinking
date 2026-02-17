import sympy as sp
x = sp.symbols('x')
expr = x**2 * sp.log(1+x)
integral = sp.integrate(expr, (x, 0, 1))
result = sp.N(integral)
print(result)