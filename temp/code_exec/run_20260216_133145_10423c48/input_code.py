import sympy as sp
x = sp.symbols('x')
f = x**2 - 2
fprime = sp.diff(f, x)
true_root = sp.N(sp.sqrt(2), 30)
x_n = 1.0
max_iter = 1000
tol = 1e-8
newton_iters = 0
newton_approx = None
while True:
    x_n_sym = sp.N(x_n, 30)
    f_val = f.subs(x, x_n_sym)
    fprime_val = fprime.subs(x, x_n_sym)
    x_next = x_n_sym - f_val / fprime_val
    if abs(x_next - true_root) < tol:
        newton_approx = float(x_next)
        break
    x_n = float(x_next)
    newton_iters += 1
    if newton_iters >= max_iter:
        break
a, b = 1.0, 2.0
bisection_iters = 0
bisection_approx = None
while True:
    c = (a + b) / 2.0
    if abs(c - float(true_root)) < tol:
        bisection_approx = c
        break
    f_c = c**2 - 2
    if (a**2 - 2) * f_c < 0:
        b = c
    else:
        a = c
    bisection_iters += 1
    if bisection_iters >= max_iter:
        break
result = newton_approx
print(result)
