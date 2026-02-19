import pandas as pd
import io

csv_data = '''a,b
c1,10
c2,20
c3,30
c4,100
c5,150
c6,10
c7,20
c8,30
c9,100
c10,150'''
# Convert to a file-like object
data_io = io.StringIO(csv_data)

# Determine total anomalies using a threshold of 100 on column b
total_anomalies = (pd.read_csv(data_io)['b'] > 100).sum()
# Reset iterator for chunked reading
data_io = io.StringIO(csv_data)

desired_rate = 0.8
threshold = 100
window_size = 2
achieved_rate = 0.0
result = 0.0

while True:
    chunk_iter = pd.read_csv(data_io, chunksize=window_size)
    detection_count = 0
    for chunk in chunk_iter:
        detection_count += (chunk['b'] > threshold).sum()
    achieved_rate = detection_count / total_anomalies if total_anomalies else 0
    if achieved_rate >= desired_rate:
        result = achieved_rate
        break
    # Adjust threshold downward to increase detection
    threshold -= 10
    if threshold < 0:
        result = achieved_rate
        break
    # Keep window_size constant for this simple example

print(result)
