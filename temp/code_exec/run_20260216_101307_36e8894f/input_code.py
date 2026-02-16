import sympy as sp
love = sp.Symbol('love')
mercy = sp.Symbol('mercy')
justice = sp.Symbol('justice')
wisdom = sp.Symbol('wisdom')
faithfulness = sp.Symbol('faithfulness')
total = love + mercy + justice + wisdom + faithfulness
total_substituted = total.subs({love:1, mercy:1, justice:1, wisdom:1, faithfulness:1})
result = total_substituted
print(result)