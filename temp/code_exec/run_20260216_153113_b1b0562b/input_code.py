import sympy as sp

times = [0, 1, 2, 3, 4]
conversions = [0, 0.8, 1.5, 2.2, 2.9]
t = sp.symbols('t')
poly = sp.interpolate(list(zip(times, conversions)), t)
dpoly = sp.diff(poly, t)
result = dpoly.subs(t, 2.5).evalf()
print(result)