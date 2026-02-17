import pandas as pd
import numpy as np
import statsmodels.api as sm
np.random.seed(0)
n=500
treatment = np.random.binomial(1,0.5,n)
covariate = np.random.normal(0,1,n)
logit = 0.5*treatment + 0.3*covariate
prob = 1/(1+np.exp(-logit))
prob = np.clip(prob,0.1,0.9)
wage = 10 + 2*treatment + 1.5*covariate + np.random.normal(0,2,n)
df = pd.DataFrame({'treatment':treatment,'wage':wage,'covariate':covariate,'prob':prob})
X = sm.add_constant(df['covariate'])
logit_model = sm.Logit(df['treatment'],X).fit(disp=False)
df['pscore'] = logit_model.predict(X)
treated = df[df['treatment']==1]
control = df[df['treatment']==0]
control_sorted = control.sort_values('pscore')
matched_control_wages = []
used = set()
for _,row in treated.iterrows():
    dists = np.abs(control_sorted['pscore'] - row['pscore'])
    idx = dists.argmin()
    if idx not in used:
        matched_control_wages.append(control_sorted.iloc[idx]['wage'])
        used.add(idx)
    else:
        for j in range(len(dists)):
            if j not in used:
                matched_control_wages.append(control_sorted.iloc[j]['wage'])
                used.add(j)
                break
ate = treated['wage'].mean() - np.mean(matched_control_wages)
result = ate
print(result)