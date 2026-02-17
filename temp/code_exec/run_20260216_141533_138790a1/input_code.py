import sympy as sp
import numpy as np
x = sp.symbols('x')
expr1 = sp.integrate(x**2, (x, 0, 1))
expr2 = sp.integrate(sp.sin(x), (x, 0, sp.pi/2))
exact = sp.simplify(expr1 + expr2)
xs = np.linspace(0, 1, 10001)
numeric1 = np.trapz(xs**2, xs)
ys = np.linspace(0, np.pi/2, 10001)
numeric2 = np.trapz(np.sin(ys), ys)
numeric_combined = numeric1 + numeric2
result = numeric_combined
print(result)