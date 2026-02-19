import sympy as sp
loading = sp.Symbol('loading')
T = sp.Symbol('T')
A = 1e5
Ea = 50e3
R = 8.314
k = A * sp.exp(-Ea/(R*T))
rate = k * loading
val = rate.subs({loading: 0.01, T: 350})
result = sp.N(val)
print(result)