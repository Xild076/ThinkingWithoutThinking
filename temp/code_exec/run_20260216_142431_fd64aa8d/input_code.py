import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import sympy as sp
np.random.seed(0)
df = pd.DataFrame({'z_score': np.random.normal(0,1,1000)})
mean_z = df['z_score'].mean()
std_z = df['z_score'].std()
refined_threshold = stats.norm.ppf(0.99, loc=mean_z, scale=std_z)
result = refined_threshold
print(result)