import pandas as pd
matrix = {"Dimension": ["Surveillance", "Biopolitics"], "1984": ["Constant monitoring", "Political control of life"], "Brave New World": ["Control through pleasure", "Genetic caste"], "The Handmaid's Tale": ["Surveillance of bodies", "Reproductive control"]}
df = pd.DataFrame(matrix)
result = df.shape[0]
print(result)