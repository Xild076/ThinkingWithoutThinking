import pandas as pd
import numpy as np
import statsmodels.api as sm

# Create synthetic data
np.random.seed(0)
n = 500
df = pd.DataFrame({
    'outcome': np.random.normal(size=n),
    'treat': np.random.choice([0,1], size=n),
    'post': np.random.choice([0,1], size=n),
    'time': np.random.randint(1,5, size=n),
    'id': np.arange(n)
})

# Placebo test
 df_placebo = df.copy()
 df_placebo['treat_shifted'] = df['treat'].shift(1)
 df_placebo = df_placebo.dropna(subset=['treat_shifted'])
 placebo_exog = sm.add_constant(df_placebo[['treat_shifted'] + [c for c in df_placebo.columns if c not in ['outcome','treat_shifted','id','time']]])
 placebo_mod = sm.OLS(df_placebo['outcome'], placebo_exog).fit()
 placebo_coef = placebo_mod.params['treat_shifted']

# Alternative specification
 df_alt = df.copy()
 df_alt['time_sq'] = df['time']**2
 alt_exog = sm.add_constant(df_alt[['treat','post','time','time_sq'] + [c for c in df_alt.columns if c not in ['outcome','treat','post','time','time_sq']]])
 alt_mod = sm.OLS(df_alt['outcome'], alt_exog).fit()
 alt_coef = alt_mod.params['treat']

# Sensitivity analysis
 sensitivity_coefs = {}
 covariates = [c for c in df.columns if c not in ['outcome','treat','post','time']]
 for cov in covariates:
     exog = sm.add_constant(df[['treat','post'] + [c2 for c2 in df.columns if c2 not in ['outcome','treat','post',cov]]])
     sens_mod = sm.OLS(df['outcome'], exog).fit()
     sensitivity_coefs[cov] = sens_mod.params['treat']

result = {
    'placebo_coefficient': placebo_coef,
    'alternative_spec_coefficient': alt_coef,
    'sensitivity_coefficients': sensitivity_coefs
}
print(result)
