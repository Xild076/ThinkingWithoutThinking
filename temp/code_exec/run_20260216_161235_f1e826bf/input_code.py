import numpy as np
import scipy.stats as stats
doc = "Safety Documentation: This document outlines safety measures for AI models."
ref = np.array([1,2,3,4,5])
curr = np.array([1.1,2.1,3.1,4.1,5.1])
drift_stat = stats.ks_2samp(ref, curr).statistic
result = drift_stat
print(result)