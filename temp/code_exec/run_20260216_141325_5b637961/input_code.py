import sympy as sp
vectors = []
for i in range(2**5):
    bits = [(i>>j)&1 for j in range(5)]
    critical = bits[:3]
    vec = sp.Matrix(bits)
    vectors.append(vec)
result = len(vectors)
print(result)