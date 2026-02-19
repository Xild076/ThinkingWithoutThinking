jobs = [(2, 100, 1), (1, 10, 2), (3, 50, 3), (3, 30, 4), (1, 20, 5)]
sorted_jobs = sorted(jobs, key=lambda x: x[1], reverse=True)
schedule = []
total_weighted_completion = 0
current_time = 0
for deadline, weight, job_id in sorted_jobs:
    if current_time + 1 <= deadline:
        schedule.append(job_id)
        current_time += 1
        total_weighted_completion += weight * current_time
result = (schedule, total_weighted_completion)
print(result)