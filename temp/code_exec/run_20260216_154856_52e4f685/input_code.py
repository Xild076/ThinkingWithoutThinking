import sympy as sp
L = sp.Rational(1,100)
D = sp.Rational(2,10**9)
observed_rate = sp.Rational(5,1000)
result = (observed_rate * L**2) / D
print(result)