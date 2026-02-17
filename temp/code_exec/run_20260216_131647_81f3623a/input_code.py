import sympy as sp
x = sp.symbols('x')
f = x**3 - x - 1
df = sp.diff(f, x)
f_func = sp.lambdify(x, f, modules='math')
df_func = sp.lambdify(x, df, modules='math')
def newton(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

def bisection(f, a, b, tol=1e-8, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise ValueError('No sign change')
    for i in range(max_iter):
        c = (a+b)/2
        fc = f(c)
        if abs(b-a) < tol or abs(fc) < tol:
            return c, i+1
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, max_iter

root_n, it_n = newton(f_func, df_func, 1.5)
root_b, it_b = bisection(f_func, 1, 2)
result = root_n
print(result)
