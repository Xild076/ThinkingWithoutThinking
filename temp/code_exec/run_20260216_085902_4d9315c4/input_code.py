
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-2,2,100)
y = x**2
plt.plot(x,y)
plt.title('y=x^2')
result = float(y.mean())
print(result)
