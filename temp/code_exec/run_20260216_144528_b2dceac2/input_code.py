import pandas as pd
import numpy as np
input_file = 'input.csv'
output_file = 'output.csv'
chunk_size = 10000
threshold = 3
result = 0
output_rows = []
reader = pd.read_csv(input_file, chunksize=chunk_size)
for chunk in reader:
    grp = chunk.groupby('entity')['value']
    mean = grp.transform(lambda s: s.rolling(10, min_periods=1).mean())
    std = grp.transform(lambda s: s.rolling(10, min_periods=1).std())
    z = (chunk['value'] - mean) / std
    anomaly = (z.abs() > threshold)
    flagged = chunk[anomaly].copy()
    flagged['anomaly'] = True
    output_rows.append(flagged)
    result += anomaly.sum()
    if len(output_rows) >= 5:
        pd.concat(output_rows).to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
        output_rows = []
if output_rows:
    pd.concat(output_rows).to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
print(result)