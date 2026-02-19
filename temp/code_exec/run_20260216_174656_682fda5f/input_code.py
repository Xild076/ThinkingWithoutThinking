import numpy as np
a = 0.0
b = np.pi
n = 100000
dx = (b - a) / n
xs = np.linspace(a, b, n, endpoint=False)
ys = np.sin(xs)
integral = np.sum(ys) * dx
result = integral
print(result)