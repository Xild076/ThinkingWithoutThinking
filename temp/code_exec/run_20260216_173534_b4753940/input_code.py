import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

# Assume df_wage and df_emp exist with columns 'treated', 'age', 'education', 'experience'
df = pd.concat([df_wage, df_emp], ignore_index=True)

covariates = ['age', 'education', 'experience']
X = sm.add_constant(df[covariates])
y = df['treated']
propensity_model = sm.Logit(y, X).fit(disp=False)
df['propensity'] = propensity_model.predict(X)

# Separate treated and control
treated = df[df['treated'] == 1]
control = df[df['treated'] == 0]

# Match each treated to nearest control on propensity
matches = []
for _, t_row in treated.iterrows():
    t_prop = t_row['propensity']
    dists = cdist([[t_prop]], control[['propensity']].values, metric='euclidean')
    nearest_idx = dists.argmin()
    matches.append((t_row.name, control.index[nearest_idx]))

# Get matched rows
matched_indices = [i for pair in matches for i in pair]
matched_df = df.loc[matched_indices].copy()
matched_df['match_id'] = (matched_df.index // 2)  # simple grouping

# Balance diagnostics: standardized mean differences
def std_mean_diff(tr, co):
    diff = tr.mean() - co.mean()
    pooled_std = ((tr.var() + co.var()) / 2) ** 0.5
    return diff / pooled_std

smds = {}
for var in covariates:
    tr_vals = matched_df.loc[matched_df['treated']==1, var]
    co_vals = matched_df.loc[matched_df['treated']==0, var]
    smds[var] = std_mean_diff(tr_vals, co_vals)

# Final result
result = {
    'matched_rows': len(matched_df),
    'average_smd': sum(smds.values()) / len(smds) if smds else 0,
    'smd_by_covariate': smds
}
print(result)
