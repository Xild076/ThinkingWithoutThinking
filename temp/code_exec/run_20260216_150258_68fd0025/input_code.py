import sympy as sp
k = sp.symbols('k', integer=True, positive=True)
n = sp.Integer(20)
S = sp.summation(k*2**k, (k,1,n))
closed = (n-1)*2**(n+1)+2
assert sp.simplify(S-closed)==0
result = sp.simplify(closed)
print(result)