import sympy as sp
import numpy as np

# closed-form expression (example: pi/2)
closed_form = sp.pi/2
# optional verification variable
verify = sp.N(closed_form, 15)

# direct numerical integration of original integral (e.g., sin(x)/x from 0 to 1000)
def f(x):
    return np.sin(x)/x if x != 0 else 1.0

xs = np.linspace(0, 1000, 200001)
ys = f(xs)
integral_val = np.trapz(ys, xs)

result = integral_val
print(result)