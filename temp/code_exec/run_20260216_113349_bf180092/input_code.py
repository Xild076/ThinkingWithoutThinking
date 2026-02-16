import sympy as sp
x = sp.Symbol('x')
analytical = sp.integrate(x**2 * sp.log(1+x), (x, 0, 1))
result = sp.N(analytical, 15)
print(result)