import sympy as sp
import numpy as np
from scipy import integrate
x = sp.symbols('x')
expr = x**2 * sp.log(1+x)
closed_form = sp.integrate(expr, (x, 0, 1))
closed_form_simplified = sp.simplify(closed_form)
numeric_closed = sp.N(closed_form_simplified, 15)
f = sp.lambdify(x, expr, 'numpy')
numeric_integral, _ = integrate.quad(lambda t: t**2 * np.log(1+t), 0, 1)
result = float(closed_form_simplified)
print(result)