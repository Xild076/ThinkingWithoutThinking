import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)**3 * (1+x)**2)
partial = sp.apart(expr, x)
result = sp.series(partial, x, 0, 11).removeO()
print(result)