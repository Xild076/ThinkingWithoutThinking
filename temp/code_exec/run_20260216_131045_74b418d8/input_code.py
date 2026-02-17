import sympy as sp
R, C, w = sp.symbols('R C w', positive=True, real=True)
H = 1 / (1 + sp.I*w*R*C)
mag = sp.simplify(sp.Abs(H))
val = {R: 1e3, C: 1e-6, w: 1e3}
result = sp.N(mag.subs(val))
print(result)