import sympy as sp

# Define symbol and function
x = sp.symbols('x')
f = sp.cos(x) - x
fp = sp.diff(f, x)

# Newton-Raphson iteration
def newton_raphson(x0, max_iter=30, tol=1e-14):
    xi = sp.N(x0, 50)  # high precision
    for i in range(max_iter):
        fi = sp.N(f.subs(x, xi), 50)
        fpi = sp.N(fp.subs(x, xi), 50)
        xi_next = xi - fi / fpi
        if abs(xi_next - xi) < tol:
            return xi_next
        xi = xi_next
    return xi

# Bisection method
def bisection(a, b, max_iter=100, tol=1e-14):
    fa = sp.N(f.subs(x, a), 50)
    fb = sp.N(f.subs(x, b), 50)
    if fa * fb > 0:
        raise ValueError("Function does not change sign on the interval")
    for i in range(max_iter):
        c = (a + b) / 2
        fc = sp.N(f.subs(x, c), 50)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fa
    return (a + b) / 2

# Compute root using Newton-Raphson starting from midpoint 0.75
result = newton_raphson(0.75)

# Optionally, also compute bisection root (not needed for final result)
# result_bisect = bisection(0.5, 1.0)

print(result)