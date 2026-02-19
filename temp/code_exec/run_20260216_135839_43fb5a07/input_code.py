values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

density = [v/w for v,w in zip(values, weights)]
items = sorted(range(len(weights)), key=lambda i: density[i], reverse=True)
greedy_schedule = []
greedy_obj = 0
for i in items:
    if weights[i] <= capacity:
        greedy_schedule.append(i)
        greedy_obj += values[i]
        capacity -= weights[i]

n = len(weights)
dp_table = [[0]*(capacity+1) for _ in range(n+1)]
for i in range(1, n+1):
    for w in range(capacity+1):
        if weights[i-1] <= w:
            dp_table[i][w] = max(dp_table[i-1][w], dp_table[i-1][w-weights[i-1]] + values[i-1])
        else:
            dp_table[i][w] = dp_table[i-1][w]
dp_obj = dp_table[n][capacity]
schedule = []
w = capacity
for i in range(n, 0, -1):
    if dp_table[i][w] != dp_table[i-1][w]:
        schedule.append(i-1)
        w -= weights[i-1]
dp_schedule = list(reversed(schedule))
structured = {
    'greedy': {'objective': greedy_obj, 'schedule': [weights[i] for i in greedy_schedule]},
    'dp': {'objective': dp_obj, 'schedule': [weights[i] for i in dp_schedule]}
}
result = structured['dp']['objective'] - structured['greedy']['objective']
print(result)