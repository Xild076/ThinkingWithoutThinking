import sympy as sp
x = sp.symbols('x')
integrand = x**2 * sp.log(1+x)
res = sp.integrate(integrand, (x, 0, 1))
result = sp.N(res)
print(result)