import sympy as sp
import numpy as np

closed_form = sp.pi/2
cf_val = sp.N(closed_form, 15)

def f(x):
    return np.where(x == 0, 1.0, np.sin(x)/x)

xs = np.linspace(0, 1000, 200001)
ys = f(xs)
integral_val = np.trapz(ys, xs)
result = integral_val
print(result)