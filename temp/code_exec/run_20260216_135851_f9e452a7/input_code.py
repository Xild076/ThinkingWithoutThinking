def greedy(target, coins):
    cnt=0
    for c in sorted(coins, reverse=True):
        while target>=c:
            target-=c
            cnt+=1
    return cnt

def dp(target, coins):
    INF=10**9
    dp=[0]+[INF]*target
    for i in range(1,target+1):
        for c in coins:
            if i>=c:
                if dp[i-c]+1<dp[i]:
                    dp[i]=dp[i-c]+1
    return dp[target]

target=6
coins=[1,3,4]
g=greedy(target, coins)
d=dp(target, coins)
diff=g-d
report=f'Greedy used {g} coins, DP used {d} coins. Greedy is faster but not always optimal; DP guarantees optimality at higher computational cost. Difference: {diff}'
result=report
print(result)