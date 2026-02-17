import numpy as np
import sympy as sp
a0 = 1.0
b0 = 2.0
a, b = sp.symbols('a b')
f = a**2 + b**3
def f_val(a_val, b_val):
    return float(f.subs({a: a_val, b: b_val}))
np.random.seed(0)
n_boot = 1000
boot_vals = []
for _ in range(n_boot):
    a_samp = np.random.normal(a0, 0.1)
    b_samp = np.random.normal(b0, 0.1)
    boot_vals.append(f_val(a_samp, b_samp))
boot_vals = np.array(boot_vals)
ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
df_da = sp.diff(f, a)
df_db = sp.diff(f, b)
sens_a = abs(float(df_da.subs({a: a0, b: b0})))
sens_b = abs(float(df_db.subs({a: a0, b: b0})))
sensitivity = sens_a + sens_b
result = ci_low
print(result)