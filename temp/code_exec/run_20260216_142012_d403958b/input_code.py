import pandas as pd
jobs = [{'id':1,'deadline':2,'weight':100},{'id':2,'deadline':1,'weight':10},{'id':3,'deadline':2,'weight':50},{'id':4,'deadline':1,'weight':70},{'id':5,'deadline':3,'weight':30}]
def greedy_schedule(jobs):
    max_deadline = max(j['deadline'] for j in jobs)
    slots = [None]*(max_deadline+1)
    schedule = []
    total_weight = 0
    for job in sorted(jobs, key=lambda j: j['weight'], reverse=True):
        for t in range(job['deadline'],0,-1):
            if slots[t] is None:
                slots[t]=job['id']
                schedule.append((job['id'],t))
                total_weight += job['weight']
                break
    return schedule,total_weight
def dp_exact(jobs):
    max_deadline = max(j['deadline'] for j in jobs)
    n = len(jobs)
    best_weight = 0
    best_schedule = []
    from itertools import combinations
    for r in range(1,n+1):
        for combo in combinations(range(n),r):
            subset = [jobs[i] for i in combo]
            subset_sorted = sorted(subset, key=lambda j: j['deadline'], reverse=True)
            slots = [None]*(max_deadline+1)
            total = 0
            schedule = []
            feasible = True
            for job in subset_sorted:
                for t in range(job['deadline'],0,-1):
                    if slots[t] is None:
                        slots[t]=job['id']
                        schedule.append((job['id'],t))
                        total += job['weight']
                        break
                else:
                    feasible = False
                    break
            if feasible and total > best_weight:
                best_weight = total
                best_schedule = schedule
    return best_schedule,best_weight
greedy_sched, greedy_weight = greedy_schedule(jobs)
dp_sched, dp_weight = dp_exact(jobs)
if greedy_weight > dp_weight:
    result = f'Greedy weight {greedy_weight} exceeds DP weight {dp_weight}'
elif greedy_weight < dp_weight:
    result = f'DP weight {dp_weight} exceeds Greedy weight {greedy_weight}'
else:
    result = f'Both weights equal {greedy_weight}'
print(result)