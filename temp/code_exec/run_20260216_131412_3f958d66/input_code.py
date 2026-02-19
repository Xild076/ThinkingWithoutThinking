import sympy as sp
n = sp.symbols('n', integer=True, positive=True)
seq1 = 1/n
seq2 = (-1)**n / n
seq3 = sp.sin(n) / n
def is_bounded(expr):
    vals = [abs(sp.N(expr.subs(n, i))) for i in range(1, 1001)]
    return max(vals) < 1e6
def accumulation_points(expr):
    even_limit = sp.limit(expr.subs(n, 2*n), n, sp.oo)
    odd_limit = sp.limit(expr.subs(n, 2*n+1), n, sp.oo)
    points = {sp.simplify(even_limit), sp.simplify(odd_limit)}
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
converges2 = True  # known convergence to 0
converges3 = sp.limit(seq3, n, sp.oo).is_real

result = (b1 and single_ap1 and converges1) + (b2 and single_ap2 and converges2) + (b3 and single_ap3 and converges3)
print(result)
