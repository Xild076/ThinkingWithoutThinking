import sympy as sp
s_values = [sp.Float(0.01), sp.Float(0.05), sp.Float(0.1)]
p0 = sp.Rational(1,100)
N = 100
final_ps = []
for s in s_values:
    p = p0
    for _ in range(N):
        p = p * (1 + s) / (p * (1 + s) + (1 - p))
    final_ps.append(p)
result = sum(final_ps) / len(final_ps)
print(result)