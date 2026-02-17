import sympy as sp
latency = sp.Integer(200)
errors = sp.Integer(5)
error_rate = errors / latency
burn_rate = (error_rate / latency) * 100
result = burn_rate
print(result)