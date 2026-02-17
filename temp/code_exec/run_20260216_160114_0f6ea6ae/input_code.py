import sympy as sp

k1, k2, k3 = sp.symbols('k1 k2 k3', positive=True)
data = {k1: 0.02, k2: 0.05, k3: 0.01}
safety_factor = k1 + k2 + k3
result = safety_factor.subs(data)
print(result)