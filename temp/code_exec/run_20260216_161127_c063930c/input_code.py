import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(0)
N = 1000
X = np.random.normal(size=(N,5))
logit = 0.5 + 0.3*X[:,0] - 0.2*X[:,1]
prob = 1/(1+np.exp(-logit))
p_treat = prob
treatment = np.random.binomial(1, p_treat, N)
Y0 = 2 + 0.5*X[:,0] + np.random.normal(scale=1, size=N)
Y1 = 5 + 0.5*X[:,0] + np.random.normal(scale=1, size=N)
Y = np.where(treatment==1, Y1, Y0)
df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
df['treatment'] = treatment
df['Y'] = Y

X_design = sm.add_constant(df[[f'x{i}' for i in range(5)]])
prop_model = sm.Logit(df['treatment'], X_design).fit(disp=False)
df['ps'] = prop_model.predict(X_design)

def match_one(treated_ps, control_ps):
    diffs = np.abs(control_ps - treated_ps)
    return control_ps[np.argmin(diffs)]

treated_idx = df[df['treatment']==1].index
control_ps = df.loc[df['treatment']==0, 'ps'].values

ate_estimates = []
for i in treated_idx:
    treated_ps = df.loc[i, 'ps']
    matched_ps = match_one(treated_ps, control_ps)
    matched_idx = df.loc[df['treatment']==0].index[np.where(np.isclose(df.loc[df['treatment']==0, 'ps'], matched_ps, atol=1e-8))[0][0]]
    ate_estimates.append(df.loc[i, 'Y'] - df.loc[matched_idx, 'Y'])

ate_mean = np.mean(ate_estimates)

X_alt = sm.add_constant(df[[f'x{i}' for i in range(1)]])
prop_model_alt = sm.Logit(df['treatment'], X_alt).fit(disp=False)
df['ps_alt'] = prop_model_alt.predict(X_alt)
treated_ps_alt = df.loc[df['treatment']==1, 'ps_alt'].values
control_ps_alt = df.loc[df['treatment']==0, 'ps_alt'].values
ate_estimates_alt = []
for i in treated_idx:
    treated_ps = df.loc[i, 'ps_alt']
    matched_ps = match_one(treated_ps, control_ps_alt)
    matched_idx = df.loc[df['treatment']==0].index[np.where(np.isclose(df.loc[df['treatment']==0, 'ps_alt'], matched_ps, atol=1e-8))[0][0]]
    ate_estimates_alt.append(df.loc[i, 'Y'] - df.loc[matched_idx, 'Y'])
ate_mean_alt = np.mean(ate_estimates_alt)

np.random.seed(1)
df_shuffled = df.copy()
df_shuffled['treatment_shuffled'] = np.random.permutation(df['treatment'])
prop_model_placebo = sm.Logit(df_shuffled['treatment_shuffled'], X_design).fit(disp=False)
df_shuffled['ps_placebo'] = prop_model_placebo.predict(X_design)
treated_ps_placebo = df_shuffled.loc[df_shuffled['treatment_shuffled']==1, 'ps_placebo'].values
control_ps_placebo = df_shuffled.loc[df_shuffled['treatment_shuffled']==0, 'ps_placebo'].values
ate_estimates_placebo = []
for i in df_shuffled.loc[df_shuffled['treatment_shuffled']==1].index:
    treated_ps = df_shuffled.loc[i, 'ps_placebo']
    matched_ps = match_one(treated_ps, control_ps_placebo)
    matched_idx = df_shuffled.loc[df_shuffled['treatment_shuffled']==0].index[np.where(np.isclose(df_shuffled.loc[df_shuffled['treatment_shuffled']==0, 'ps_placebo'], matched_ps, atol=1e-8))[0][0]]
    ate_estimates_placebo.append(df_shuffled.loc[i, 'Y'] - df_shuffled.loc[matched_idx, 'Y'])
ate_mean_placebo = np.mean(ate_estimates_placebo)

robustness_metric = np.mean([ate_mean, ate_mean_alt, ate_mean_placebo])
result = robustness_metric
print(result)
