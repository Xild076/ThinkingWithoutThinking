import pandas as pd

chunk_size = 10000
file_path = 'large_file.csv'
total = 0
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    total += chunk['value'].sum()
result = total
print(result)