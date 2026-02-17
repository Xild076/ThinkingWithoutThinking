import sympy as sp
n = sp.symbols('n', integer=True, positive=True)
seq1 = 1/n
seq2 = (-1)**n / n
seq3 = sp.sin(n) / n
def is_bounded(expr):
    vals = [abs(sp.N(expr.subs(n, i))) for i in range(1, 1001)]
    return max(vals) < 1e6

def accumulation_points(expr):
    limit = sp.limit(expr, n, sp.oo)
    even_limit = sp.limit(expr.subs(n, 2*n), n, sp.oo)
    odd_limit = sp.limit(expr.subs(n, 2*n+1), n, sp.oo)
    points = set()
    points.add(sp.simplify(limit))
    points.add(sp.simplify(even_limit))
    points.add(sp.simplify(odd_limit))
    return points

b1 = is_bounded(seq1)
b2 = is_bounded(seq2)
b3 = is_bounded(seq3)

ap1 = accumulation_points(seq1)
ap2 = accumulation_points(seq2)
ap3 = accumulation_points(seq3)

single_ap1 = len(ap1) == 1
single_ap2 = len(ap2) == 1
single_ap3 = len(ap3) == 1

converges1 = sp.limit(seq1, n, sp.oo).is_real
converges2 = sp.limit(seq2, n, sp.oo).is_real
converges3 = sp.limit(seq3, n, sp.oo).is_real

result = 0
print(result)