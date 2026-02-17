import math

y = 1
while True:
    x2 = 5*y*y + 1
    x = math.isqrt(x2)
    if x*x == x2:
        result = (x, y)
        break
    y += 1
print(result)