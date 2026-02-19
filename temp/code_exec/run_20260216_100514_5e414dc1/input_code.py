import sympy as sp
x = sp.symbols('x')
expr = 1/((1-x)**3 * (1+x)**2)
decomp = sp.apart(expr, x)
result = decomp
print(result)