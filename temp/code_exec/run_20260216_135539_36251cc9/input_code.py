import mpmath as mp
import sympy as sp
mp.mp.dps = 30
analytic = (sp.Rational(2,3)*sp.log(2) - sp.Rational(5,18))
analytic_val = float(analytic.evalf())
num_int = mp.quad(lambda t: t**2 * mp.log(1+t), [0,1])
result = abs(num_int - analytic_val)
print(result)