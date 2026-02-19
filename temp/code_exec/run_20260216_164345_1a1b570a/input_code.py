import pandas as pd
import sympy as sp
from io import StringIO

csv_data = 'value\n10\n12\n12\n13\n12\n11\n100\n11\n12\n13\n12\n11\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1'

df = pd.read_csv(StringIO(csv_data))
values = df['value'].tolist()
sym_vals = [sp.nsimplify(v) for v in values]
mean_sym = sp.N(sp.mean(sym_vals))
std_sym = sp.N(sp.sqrt(sp.mean([(v - mean_sym)**2 for v in sym_vals])))
z_scores = [float(((v - mean_sym) / std_sym).evalf()) for v in sym_vals]
anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 3]
result = len(anomalies)
print(result)