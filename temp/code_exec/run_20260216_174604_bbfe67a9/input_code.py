import numpy as np

def integrand(x):
    return np.log(np.sin(x))

N = 2000000

a = 0.0
b = np.pi / 2
xs = np.linspace(a, b, N + 1)
ys = integrand(xs)
approx = np.trapz(ys, xs)
analytical = -np.pi / 2 * np.log(2)
diff = abs(approx - analytical)
result = f'approx={approx}, analytical={analytical}, diff={diff}'
print(result)