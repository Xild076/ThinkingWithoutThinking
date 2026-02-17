import sympy as sp
import math
f=lambda x: x**3-2
fp=lambda x: 3*x**2
alpha=sp.N(2**(sp.Rational(1,3)),50)
alpha_f=float(alpha)
x=1.5
while True:
    x_new=x-f(x)/fp(x)
    if abs(x_new-x)<1e-8:
        break
    x=x_new
error_newton=abs(x_new-alpha_f)
a,b=1.0,2.0
tol=1e-8
n=math.ceil(math.log2((b-a)/tol))
for _ in range(n):
    c=(a+b)/2.0
    if f(a)*f(c)<=0:
        b=c
    else:
        a=c
mid=(a+b)/2.0
error_bisection=abs(mid-alpha_f)
result=max(error_newton,error_bisection)
print(result)