import sympy as sp
k = sp.symbols('k', integer=True, positive=True)
deriv1 = sp.summation(1/k**2, (k, 1, 20))
deriv2 = sum([sp.Rational(1, k**2) for k in range(1, 21)])
S_20 = deriv1
consistent = deriv1.equals(deriv2)
result = f"Derivation 1 (sympy summation) = {deriv1}, Derivation 2 (list comprehension) = {deriv2}; both equal S_20 = {S_20}, consistency: {consistent}"
print(result)