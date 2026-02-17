import sympy as sp

# Define symbol and function
x = sp.symbols('x')
f = sp.exp(x) - 2
df = sp.diff(f, x)

# Newton-Raphson function
def newton_raphson(func, dfunc, x0, tol=1e-12, max_iter=100):
    x_val = x0
    for i in range(max_iter):
        fx = func.subs(x, x_val)
        dfx = dfunc.subs(x, x_val)
        x_new = x_val - fx / dfx
        if abs(x_new - x_val) < tol:
            return x_new, i+1
        x_val = x_new
    return x_val, max_iter

# Bisection function
def bisection(func, a, b, tol=1e-12, max_iter=100):
    fa = func.subs(x, a)
    fb = func.subs(x, b)
    if fa * fb > 0:
        raise ValueError('No sign change')
    for i in range(max_iter):
        c = (a + b) / 2
        fc = func.subs(x, c)
        if abs(b - a) < tol or abs(fc) < tol:
            return c, i+1
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fa
    return c, max_iter

# Compute approximations
newton_approx, newton_iters = newton_raphson(f, df, 0.7)
bisect_approx, bisect_iters = bisection(f, 0.6, 0.8)

# True root
true_root = sp.N(sp.log(2), 30)

# Errors
newton_error = abs(newton_approx - true_root)
bisect_error = abs(bisect_approx - true_root)

# Ensure errors are below 1e-8
assert newton_error < 1e-8, f'Newton error {newton_error} not below 1e-8'
assert bisect_error < 1e-8, f'Bisection error {bisect_error} not below 1e-8'

# Store result and print
result = newton_approx
print(result)
