import sympy as sp
sens = sp.Rational(47,50)
spec = sp.Rational(91,100)
prev = sp.Rational(1,50)
ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev))
npv = (spec*(1-prev)) / ((1-sens)*prev + spec*(1-prev))
result = {'PPV': ppv, 'NPV': npv}
print(result)