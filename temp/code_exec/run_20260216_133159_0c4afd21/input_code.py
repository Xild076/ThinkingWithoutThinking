import sympy as sp
import numpy as np

# define function
x = sp.symbols('x')
f = x**3 - 2
fp = sp.diff(f, x)

# convert to lambda for numeric evaluation
f_lam = sp.lambdify(x, f, 'numpy')
fp_lam = sp.lambdify(x, fp, 'numpy')

# Newton-Raphson
x_n = 1.0
newton_errors = []
while True:
    fx = f_lam(x_n)
    err = abs(fx)
    newton_errors.append(err)
    if err < 1e-8:
        break
    x_n = x_n - fx / fp_lam(x_n)

# Bisection
a, b = 1.0, 2.0
bisection_errors = []
while True:
    c = (a + b) / 2
    fc = f_lam(c)
    err = abs(fc)
    bisection_errors.append(err)
    if err < 1e-8:
        break
    if f_lam(a) * fc < 0:
        b = c
    else:
        a = c

# compute iteration counts
newton_iters = len(newton_errors)
bisection_iters = len(bisection_errors)
result = newton_iters - bisection_iters
print(result)
