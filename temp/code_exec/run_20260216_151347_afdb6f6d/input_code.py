import sympy as sp
C1, C2 = sp.symbols('C1 C2')
sol = sp.solve([C1 + 4 - 1, (C1 + C2)*3 + 8 - 4], (C1, C2))
result = (sol[C1], sol[C2])
print(result)