import pandas as pd
import sympy as sp
from io import StringIO
from scipy import stats

csv_data = """value
10
12
12
13
12
11
100
11
12
13
12
11
10
9
8
7
6
5
4
3
2
1"""

df = pd.read_csv(StringIO(csv_data))

col = df['value']
sym_vals = col.apply(lambda x: sp.nsimplify(x))

mean_sym = sp.N(sym_vals.mean())
std_sym = sp.N(sym_vals.std(ddof=0))

z_sym = [(x - mean_sym) / std_sym if std_sym != 0 else 0 for x in sym_vals]
z_scores = [float(z) for z in z_sym]

anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 3]

summary = f'Total rows: {len(z_scores)}, Anomalies detected: {len(anomalies)}'
result = len(anomalies)
print(result)
