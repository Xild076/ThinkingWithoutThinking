import sympy as sp
import numpy as np
from scipy import integrate

x = sp.symbols('x')
symbolic_val = sp.integrate(x**2 * sp.log(1+x), (x, 0, 1))
numeric_val, _ = integrate.quad(lambda t: t**2 * np.log(1+t), 0, 1)
result = {"numeric": numeric_val, "symbolic": sp.N(symbolic_val, 15)}
print(result)