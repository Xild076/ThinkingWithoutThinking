import pandas as pd
treated_post = df[(df['treated']==1) & (df['post']==1)]
treated_pre = df[(df['treated']==1) & (df['post']==0)]
control_post = df[(df['treated']==0) & (df['post']==1)]
control_pre = df[(df['treated']==0) & (df['post']==0)]

effect = (treated_post['employment'].mean() - treated_pre['employment'].mean()) - (control_post['employment'].mean() - control_pre['employment'].mean())

result = effect
print(result)