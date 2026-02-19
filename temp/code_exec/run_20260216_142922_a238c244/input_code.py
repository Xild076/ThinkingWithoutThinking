n = 20
total = 0
for k in range(1, n+1):
    total += k * (2**k)
result = total
closed_form = (n-1)*(2**(n+1)) + 2
print(result)