import sympy as sp
import numpy as np
x=sp.symbols('x')
exact=sp.integrate(x**2*sp.log(1+x),(x,0,1))
result=sp.N(exact,15)
print(result)