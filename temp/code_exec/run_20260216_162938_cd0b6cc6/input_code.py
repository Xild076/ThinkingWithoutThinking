import pandas as pd
import tempfile
import os

# Create a sample CSV file with some data
data = 'id,value\\n' + '\\n'.join([f'{i},{i*2}' for i in range(1, 10001)])
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
    f.write(data)
    temp_path = f.name

# Define chunk size
chunk_size = 5000

total = 0
for chunk in pd.read_csv(temp_path, chunksize=chunk_size):
    total += chunk['value'].sum()

result = total
print(result)

# Clean up temporary file
os.remove(temp_path)
