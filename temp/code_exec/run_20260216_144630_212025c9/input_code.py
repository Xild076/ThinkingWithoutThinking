import pandas as pd
import random
n = 1000000
col_a = [random.random() for _ in range(n)]
col_b = [random.randint(0, 1000) for _ in range(n)]
col_c = [random.choice(['X','Y','Z']) for _ in range(n)]
df = pd.DataFrame({'A': col_a, 'B': col_b, 'C': col_c})
df.to_csv('synthetic.csv', index=False)
df2 = pd.read_csv('synthetic.csv')
assert df2.shape == (n, 3)
mem_bytes = df2.memory_usage(deep=True).sum()
result = mem_bytes
print(result)