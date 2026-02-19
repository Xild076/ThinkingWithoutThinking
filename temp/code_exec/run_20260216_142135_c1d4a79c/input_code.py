import pandas as pd

n = 0
mean = 0.0
M2 = 0.0
for chunk in pd.read_csv('data.csv', chunksize=10000):
    for _, row in chunk.iterrows():
        x = row['value']
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
if n > 0:
    result = mean
else:
    result = 0.0
print(result)