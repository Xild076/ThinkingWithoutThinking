import pandas as pd
from itertools import permutations


data = {
    'p': [1,1,1,2,1],
    'd': [2,3,4,4,5],
    'w': [3,2,2,1,4]
}
df = pd.DataFrame(data)
n = len(df)

sorted_idx = sorted(range(n), key=lambda i: -df.loc[i, 'w'])
schedule_greedy = []
time_greedy = 0
cost_greedy = 0
for i in sorted_idx:
    p = df.loc[i, 'p']
    d = df.loc[i, 'd']
    w = df.loc[i, 'w']
    if time_greedy + p <= d:
        schedule_greedy.append(i)
        time_greedy += p
        cost_greedy += w * time_greedy

from functools import lru_cache
@lru_cache(None)
def dp(mask, t):
    if mask == 0:
        return 0
    best = None
    for i in range(n):
        if mask & (1 << i):
            p = df.loc[i, 'p']
            d = df.loc[i, 'd']
            w = df.loc[i, 'w']
            if t + p <= d:
                new_mask = mask ^ (1 << i)
                cost = w * (t + p) + dp(new_mask, t + p)
                if best is None or cost < best:
                    best = cost
    return best if best is not None else float('inf')

optimal_cost = dp((1 << n) - 1, 0)

diff = cost_greedy - optimal_cost
result = diff
print(result)
