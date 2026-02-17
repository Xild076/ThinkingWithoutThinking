import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import sympy as sp
np.random.seed(0)
n=1000
X=np.random.normal(size=(n,2))
treatment=(X[:,0]+X[:,1]>0).astype(int)
wage=10+2*treatment+np.random.normal(scale=1,size=n)
df=pd.DataFrame({'X1':X[:,0],'X2':X[:,1],'treatment':treatment,'wage':wage})
X_design=sm.add_constant(df[['X1','X2']])
propensity=sm.Logit(df['treatment'],X_design).fit(disp=False).predict()
df['ps']=propensity
try:
    from sklearn.neighbors import NearestNeighbors
    nn=NearestNeighbors(n_neighbors=1,algorithm='ball_tree')
nn.fit(df.loc[df['treatment']==0,['ps']])
distances,indices=nn.kneighbors(df.loc[df['treatment']==1,['ps']],n_neighbors=1)
matches=df.loc[df['treatment']==0].iloc[indices.ravel()]
matches=matches.reset_index(drop=True)
diff=matches['wage'].values - df.loc[df['treatment']==1,'wage'].values[:len(matches)]
result=diff.mean()
except Exception:
    treated_ps=df.loc[df['treatment']==1,'ps'].values
    control_ps=df.loc[df['treatment']==0,'ps'].values
    control_wages=df.loc[df['treatment']==0,'wage'].values
    ate_sum=0.0
    count=0
    for ps in treated_ps:
        idx=np.argmin(np.abs(control_ps-ps))
        ate_sum+=df.loc[df['treatment']==1,'wage'].iloc[count]-control_wages[idx]
        count+=1
    result=ate_sum/len(treated_ps) if len(treated_ps)>0 else 0.0
print(result)