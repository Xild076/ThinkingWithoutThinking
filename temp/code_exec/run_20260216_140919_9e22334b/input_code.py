import sympy as sp
failures = [{'case': 'division_by_zero', 'expr': '1/0'}, {'case': 'invalid_syntax', 'expr': 'if True:'}, {'case': 'type_error', 'expr': 'len(5)'}]
docs = {}
for f in failures:
    docs[f['case']] = f['expr']
result_val = 0
for case, expr in failures:
    try:
        sym_expr = sp.sympify(expr)
        result_val += 1
    except Exception:
        pass
result = result_val
print(result)