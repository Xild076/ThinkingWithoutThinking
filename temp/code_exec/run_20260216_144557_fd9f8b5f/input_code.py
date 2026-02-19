import pandas as pd
import numpy as np
from pathlib import Path

input_file = 'input.csv'
output_file = 'output.csv'
chunk_size = 10000
threshold = 3
result = 0
output_rows = []
flush_limit = 1000
header_written = False

# Ensure input file exists
if not Path(input_file).exists():
    sample = pd.DataFrame({
        'entity': ['A','A','A','B','B','B'],
        'value': [1,2,3,10,11,12]
    })
    sample.to_csv(input_file, index=False)

reader = pd.read_csv(input_file, chunksize=chunk_size)
for chunk in reader:
    grp = chunk.groupby('entity')['value']
    mean = grp.transform(lambda s: s.rolling(window=10, min_periods=1).mean())
    std = grp.transform(lambda s: s.rolling(window=10, min_periods=1).std())
    std = std.replace(0, np.nan)
    z = (chunk['value'] - mean) / std
    anomaly = z.abs() > threshold
    flagged = chunk[anomaly].copy()
    flagged['anomaly'] = True
    output_rows.append(flagged)
    result += anomaly.sum()
    if len(output_rows) >= flush_limit:
        pd.concat(output_rows).to_csv(output_file, mode='a', header=not header_written, index=False)
        output_rows = []
        header_written = True
if output_rows:
    pd.concat(output_rows).to_csv(output_file, mode='a', header=header_written, index=False)

print(result)
