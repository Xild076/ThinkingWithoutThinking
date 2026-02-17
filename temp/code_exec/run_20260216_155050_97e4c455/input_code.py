import pandas as pd
import statsmodels.formula.api as smf

# create synthetic panel data
df = pd.DataFrame({
    'group': ['A','A','A','A','B','B','B','B'],
    'time': [1,2,3,4,1,2,3,4],
    'employment': [100,101,102,103,99,100,101,102],
    'min_wage': [7.25,7.25,7.5,7.5,7.25,7.25,7.25,7.25]
})

# post indicator (treatment period)
df['post'] = (df['time'] >= 3).astype(int)
# treated group indicator
df['treated'] = (df['group'] == 'A').astype(int)
# interaction term
df['interaction'] = df['post'] * df['treated']

# run regression
model = smf.ols('employment ~ post + treated + interaction', data=df).fit()
did_coef = model.params['interaction']
result = did_coef
print(result)