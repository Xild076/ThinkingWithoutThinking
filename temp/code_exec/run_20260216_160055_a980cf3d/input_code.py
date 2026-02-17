import sympy as sp
import numpy as np

# Define symbols
D, k, L, t_end = sp.symbols('D k L t_end', positive=True, real=True)

# Define observable expressions
obs_diff = sp.exp(-t_end * D / L**2)
obs_rxn = 1 - sp.exp(-k * t_end)
obs = (obs_diff + obs_rxn) / 2

# Create a numeric function
f = sp.lambdify((D, k, L, t_end), obs, modules='numpy')

# Sampling parameters
rng = np.random.default_rng(seed=0)
samples = 1000
L_val = 1.0
t_end_val = 1.0
D_samples = rng.normal(loc=0.5, scale=0.1, size=samples)
k_samples = rng.normal(loc=0.8, scale=0.2, size=samples)

# Compute observables and final result
obs_vals = f(D_samples, k_samples, L_val, t_end_val)
result = np.mean(obs_vals)

print(result)
