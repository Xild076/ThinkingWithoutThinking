import sympy as sp

def newton_raphson(f, df, x0, tol=1e-12, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = float(f.subs(x, x))
        dfx = float(df.subs(x, x))
        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

def bisection(f, a, b, tol=1e-12, max_iter=100):
    fa = float(f.subs(x, a))
    fb = float(f.subs(x, b))
    if fa*fb > 0:
        raise ValueError('No sign change')
    for i in range(max_iter):
        c = (a+b)/2
        fc = float(f.subs(x, c))
        if abs(b-a) < tol or abs(fc) < tol:
            return c, i+1
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fa
    return c, max_iter

x = sp.symbols('x')
f = sp.exp(x) - 2
 df = sp.diff(f, x)

newton_approx, newton_iters = newton_raphson(f, df, 0.7)
bisect_approx, bisect_iters = bisection(f, 0.6, 0.8)

true_root = float(sp.N(sp.log(2), 30))

newton_error = abs(newton_approx - true_root)
bisect_error = abs(bisect_approx - true_root)

result = newton_approx
print(result)
