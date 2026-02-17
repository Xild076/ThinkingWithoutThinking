import sympy as sp
import numpy as np

x = sp.symbols('x')
I1 = sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))
I2 = sp.integrate(sp.exp(-x), (x, 0, sp.oo))
combined = sp.simplify(I1 + I2)
numeric_check = sp.N(combined, 20)
result = numeric_check
print(result)