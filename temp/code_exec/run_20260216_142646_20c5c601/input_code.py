import numpy as np
import pandas as pd
np.random.seed(0)
n=1000
age=np.random.randint(18,80,size=n)
severity=np.random.rand(n)*10
utility=(age/100)*severity
result=utility.mean()
print(result)