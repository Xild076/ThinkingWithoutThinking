import math
from scipy import integrate

def integrand(x):
    return math.log(math.sin(x))

approx, _ = integrate.quad(integrand, 0, math.pi/2)
analytical = -math.pi/2 * math.log(2)
diff = abs(approx - analytical)
result = "approx={}, analytical={}, diff={}".format(approx, analytical, diff)
print(result)