import pandas as pd
import numpy as np
import statsmodels.api as sm

# generate synthetic data
np.random.seed(0)
n=1000
X=np.random.normal(size=(n,2))
treatment=(X[:,0]+X[:,1]>0).astype(int)
wage=10+2*treatment+np.random.normal(scale=1,size=n)
df=pd.DataFrame({'X1':X[:,0],'X2':X[:,1],'treatment':treatment,'wage':wage})
# estimate propensity score using logistic regression
X_design=sm.add_constant(df[['X1','X2']])
propensity=sm.Logit(df['treatment'],X_design).fit(disp=False).predict()
df['ps']=propensity

# separate treated and control
treated_idx=df[df['treatment']==1].index
control_df=df[df['treatment']==0]

# nearest neighbor matching on propensity score
ps_treated=df.loc[treated_idx,'ps'].values
ps_control=control_df['ps'].values
# find nearest control for each treated unit
nearest_idx=np.argmin(np.abs(ps_control.values[:,None]-ps_treated.values),axis=0)
matched_control_wages=control_df.iloc[nearest_idx]['wage'].values

# compute average treatment effect estimate
ate_est=(df.loc[treated_idx,'wage'].values-matched_control_wages).mean()
result=ate_est
print(result)
