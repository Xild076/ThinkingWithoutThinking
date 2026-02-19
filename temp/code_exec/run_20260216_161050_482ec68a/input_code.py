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
# propensity scores
ps_treated=df.loc[treated_idx,'ps'].to_numpy()
ps_control=control_df['ps'].to_numpy()
# matching
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None
if NearestNeighbors is not None:
    nbrs=NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(ps_control.reshape(-1,1))
    _,indices=nbrs.kneighbors(ps_treated.reshape(-1,1))
    nearest_idx=indices.ravel()
else:
    distances=np.abs(ps_control[:,None]-ps_treated)
    nearest_idx=np.argmin(distances,axis=0)
matched_control_wages=control_df.iloc[nearest_idx]['wage'].to_numpy()
# compute ATE estimate
ate_est=(df.loc[treated_idx,'wage'].to_numpy()-matched_control_wages).mean()
result=ate_est
print(result)
