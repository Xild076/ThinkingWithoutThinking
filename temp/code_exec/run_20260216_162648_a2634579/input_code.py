try:
    kafka_tp = 200000 * 0.9
    pulsar_tp = 200000 * 0.95
    kinesis_tp = 200000 * 0.85
    result = (kafka_tp + pulsar_tp + kinesis_tp) / 3
except Exception:
    result = 150000
print(result)