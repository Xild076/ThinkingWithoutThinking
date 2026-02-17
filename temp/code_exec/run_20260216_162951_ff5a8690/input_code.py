import pandas as pd
csv_data = "id,value\n" + "\n".join([f"{i},"{i*2}" for i in range(1, 10001)])
with open("data.csv", "w") as f:
    f.write(csv_data)
chunk_size = 5000
total = 0
for chunk in pd.read_csv("data.csv", chunksize=chunk_size):
    total += chunk["value"].sum()
result = total
print(result)