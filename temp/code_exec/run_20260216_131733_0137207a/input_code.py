import sympy as sp
import numpy as np

x = sp.symbols('x')
f = x**3 - 2
df = sp.diff(f, x)

f_num = sp.lambdify(x, f, 'numpy')
df_num = sp.lambdify(x, df, 'numpy')

def newton_raphson(func, df_func, x0, tol=1e-12, max_iter=100):
    x_curr = x0
    errors = []
    for i in range(max_iter):
        fx = func(x_curr)
        dfx = df_func(x_curr)
        if dfx == 0:
            break
        x_next = x_curr - fx/dfx
        err = abs(x_next - x_curr)
        errors.append(err)
        if err < tol:
            return x_next, i+1, errors
        x_curr = x_next
    return x_curr, max_iter, errors

def bisection(func, a, b, tol=1e-12, max_iter=100):
    fa = func(a)
    fb = func(b)
    if fa*fb > 0:
        raise ValueError('Function has same sign at endpoints')
    errors = []
    for i in range(max_iter):
        c = (a+b)/2
        fc = func(c)
        errors.append(abs(fc))
        if abs(fc) < tol:
            return c, i+1, errors
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a+b)/2, max_iter, errors

# parameters
x0 = 1.0
newton_root, newton_iters, newton_errs = newton_raphson(f_num, df_num, x0)
bisect_root, bisect_iters, bisect_errs = bisection(f_num, 1, 2)

def error_reduction(errors):
    if len(errors) < 2:
        return None
    return errors[-1]/errors[-2]

newton_red = error_reduction(newton_errs)
bisect_red = error_reduction(bisect_errs)

newton_fe = newton_iters + 1
bisect_fe = bisect_iters + 1

report = {
    'newton_root': newton_root,
    'newton_iters': newton_iters,
    'newton_error_reduction': newton_red,
    'newton_fe': newton_fe,
    'bisection_root': bisect_root,
    'bisection_iters': bisect_iters,
    'bisection_error_reduction': bisect_red,
    'bisection_fe': bisect_fe
}

result = report['newton_root']
print(result)
