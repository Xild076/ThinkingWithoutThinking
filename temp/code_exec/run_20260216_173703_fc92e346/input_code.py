import pandas as pd
import numpy as np
import statsmodels.api as sm

df_placebo = df.copy()
df_placebo['treat_shifted'] = df['treat'].shift(1)
placebo_exog = sm.add_constant(df_placebo[['treat_shifted'] + [c for c in df_placebo.columns if c not in ['outcome','treat_shifted','id','time']]])
placebo_mod = sm.OLS(df_placebo['outcome'], placebo_exog).fit()
placebo_coef = placebo_mod.params['treat_shifted']

df_alt = df.copy()
df_alt['time_sq'] = df['time']**2
alt_exog = sm.add_constant(df_alt[['treat','post','time','time_sq'] + [c for c in df_alt.columns if c not in ['outcome','treat','post','time','time_sq']]])
alt_mod = sm.OLS(df_alt['outcome'], alt_exog).fit()
alt_coef = alt_mod.params['treat']

sensitivity_coefs = {}
for cov in [c for c in df.columns if c not in ['outcome','treat','post','time']]:
    exog = sm.add_constant(df[['treat','post'] + [c2 for c2 in df.columns if c2 not in ['outcome','treat','post',cov]]])
    sens_mod = sm.OLS(df['outcome'], exog).fit()
    sensitivity_coefs[cov] = sens_mod.params['treat']

result = {
    'placebo_coefficient': placebo_coef,
    'alternative_spec_coefficient': alt_coef,
    'sensitivity_coefficients': sensitivity_coefs
}
print(result)