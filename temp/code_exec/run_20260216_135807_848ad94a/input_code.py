import pandas as pd

jobs = pd.DataFrame({
    'job_id': [1, 2, 3, 4, 5],
    'start': [1, 2, 3, 3, 6],
    'finish': [3, 5, 6, 8, 8],
    'profit': [50, 20, 70, 60, 40]
})
jobs = jobs.sort_values('finish').reset_index(drop=True)
n = len(jobs)

def latest_non_conflict(j):
    for i in range(j-1, -1, -1):
        if jobs.at[i, 'finish'] <= jobs.at[j, 'start']:
            return i
    return -1

dp = [0] * n
dp[0] = jobs.at[0, 'profit']

for i in range(1, n):
    include = jobs.at[i, 'profit']
    l = latest_non_conflict(i)
    if l != -1:
        include += dp[l]
    exclude = dp[i-1]
    dp[i] = max(include, exclude)

result = dp[-1]
print(result)