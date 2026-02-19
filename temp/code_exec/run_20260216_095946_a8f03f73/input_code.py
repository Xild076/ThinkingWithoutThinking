import sympy as sp
x = sp.symbols('x')
result = sp.integrate(x**2 * sp.log(1+x), (x, 0, 1))
print(result)