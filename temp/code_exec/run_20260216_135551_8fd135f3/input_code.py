result = 0.0
N = 10**6
for n in range(1, N+1):
    term = (-1)**(n+1) / (n * (n+3))
    result += term
print(result)