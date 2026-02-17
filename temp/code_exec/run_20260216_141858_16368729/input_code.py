import pandas as pd
from itertools import permutations

# Sample dataset
 data = {
     'p': [1,1,1,2,1],
     'd': [2,3,4,4,5],
     'w': [3,2,2,1,4]
 }
 df = pd.DataFrame(data)
 n = len(df)

# Greedy heuristic (schedule by weight descending, respecting deadlines)
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

# Exact DP by enumerating all permutations
 best_cost = None
 best_schedule = None
 for perm in permutations(range(n)):
     time = 0
     feasible = True
     cost = 0
     for i in perm:
         p = df.loc[i, 'p']
         d = df.loc[i, 'd']
         w = df.loc[i, 'w']
         time += p
         if time > d:
             feasible = False
             break
         cost += w * time
     if feasible:
         if best_cost is None or cost < best_cost:
             best_cost = cost
             best_schedule = perm

# Compute final result
diff = cost_greedy - best_cost
result = diff
print(result)
