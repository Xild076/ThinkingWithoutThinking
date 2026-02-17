def newton_raphson(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return i+1, x_new
        x = x_new
    return max_iter, x

def bisection(f, a, b, tol=1e-8, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise ValueError('No sign change')
    for i in range(max_iter):
        c = (a+b)/2
        fc = f(c)
        if abs(b-a) < tol or abs(fc) < tol:
            return i+1, c
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fa
    return max_iter, (a+b)/2

f = lambda x: x**3 - x - 2
 df = lambda x: 3*x**2 - 1
newton_iter, newton_root = newton_raphson(f, df, 1.0)
bisection_iter, bisection_root = bisection(f, 1.0, 2.0)
result = (newton_iter, bisection_iter)
print(result)