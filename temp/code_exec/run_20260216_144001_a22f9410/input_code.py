jobs = [(1,3,5),(2,5,6),(4,6,5),(6,7,4)]
jobs_sorted = sorted(jobs, key=lambda x: x[1])
greedy_weight = 0
selected = []
for s,f,w in jobs_sorted:
    if not selected or s >= selected[-1][1]:
        selected.append((s,f,w))
        greedy_weight += w
n = len(jobs)
jobs_idx = list(range(n))
jobs_idx.sort(key=lambda i: jobs[i][1])
dp = [0]*n
for i in jobs_idx:
    s,f,w = jobs[i]
    incl = w
    for j in jobs_idx:
        if j >= i: break
        sj,fj,wj = jobs[j]
        if fj <= s:
            incl = max(incl, w + dp[j])
    dp[i] = incl
opt_weight = max(dp)
result = opt_weight - greedy_weight
print(result)