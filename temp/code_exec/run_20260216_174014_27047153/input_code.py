import pika, time, random, json
NUM_MESSAGES = 500
latencies = []
errors = 0
conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
ch = conn.channel()
ch.exchange_declare(exchange='events', exchange_type='fanout')
for i in range(NUM_MESSAGES):
event = {'id': i, 'data': random.random()}
body = json.dumps(event)
start = time.time()
try:
    ch.basic_publish(exchange='events', routing_key='', body=body)
elapsed = (time.time() - start) * 1000  # ms
latencies.append(elapsed)
except Exception:
errors += 1
conn.close()
avg_latency = sum(latencies) / len(latencies) if latencies else 0
error_rate = errors / NUM_MESSAGES * 100
result = {"latency_ms": round(avg_latency, 2), "error_rate_percent": round(error_rate, 2)}
print(result)