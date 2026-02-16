import sympy as sp
x=sp.symbols('x')
expr=1/((1-x)**3*(1+x)**2)
series1=sp.series(expr, x, 0, 11).removeO()
coeff1=sp.expand(series1).coeff(x,10)
pf=sp.apart(expr, x)
coeff2=0
for term in pf.as_ordered_terms():
    s=sp.series(term, x, 0, 11).removeO()
    coeff2+=sp.expand(s).coeff(x,10)
assert coeff1==coeff2
result=coeff1
print(result)