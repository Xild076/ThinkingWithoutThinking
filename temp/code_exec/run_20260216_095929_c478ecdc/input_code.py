import sympy as sp
x = sp.symbols('x')
integral = sp.integrate(x**2*sp.log(1+x), (x, 0, 1))
result = sp.simplify(integral)
print(result)