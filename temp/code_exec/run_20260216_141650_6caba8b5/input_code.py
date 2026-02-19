import sympy as sp
ref = 1.4142135623730951
result = sp.N(sp.sqrt(2))
assert abs(result - ref) < 1e-12
print(result)