import mpmath as mp
f = lambda x: x**2 * mp.log(1+x)
val = mp.quad(f, [0,1])
s = mp.nsum(lambda n: (-1)**(n+1) / (n**3 * (n+1)), [1, mp.inf])
result = val
print('Numeric integral:', val)
print('Series cross-check:', s)
print(result)