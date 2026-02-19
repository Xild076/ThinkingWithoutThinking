total_events = 200_000
safety_factor = 1.2
required_throughput = total_events * safety_factor
partitions = 3
consumer_groups = 2
required_per_partition = required_throughput // (partitions * consumer_groups)
result = required_per_partition
print(result)