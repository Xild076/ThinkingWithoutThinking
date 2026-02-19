import scipy.stats as stats
import statsmodels.api as sm
sample1 = [5,6,7,8,9]
sample2 = [6,7,8,9,10]
t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
perf = 1 - p_val
safety = 0.95
completeness = 1.0
result = perf * safety * completeness
print(result)