import sympy as sp
p1 = sp.Rational(1,4)
p2 = sp.Rational(1,3)
p3 = sp.Rational(1,2)
result = p1 + p2 + p3 - p1*p2 - p1*p3 - p2*p3 + p1*p2*p3
print(result)