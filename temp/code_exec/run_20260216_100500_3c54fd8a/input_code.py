def convolve(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            c[i + j] += ai * bj
    return c[10] if len(c) > 10 else None

a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5, 6]
result = convolve(a, b)
print(result)