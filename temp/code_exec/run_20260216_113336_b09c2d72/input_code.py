import sympy as sp
import mpmath as mp

mp.mp.dps = 30

x = sp.symbols('x')
closed_form_expr = sp.integrate(x**2 * sp.exp(x), (x, 0, 1))
closed_form_val = sp.N(closed_form_expr, 30)
closed_form_mp = mp.mpf(str(closed_form_val))

f = lambda t: t**2 * mp.e**t
numeric_integral = mp.quad(f, [0, 1])

diff = abs(closed_form_mp - numeric_integral)
result = diff
print(result)
