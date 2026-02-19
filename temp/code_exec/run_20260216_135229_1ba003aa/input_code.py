import sympy as sp
def test_one():
    return sp.Rational(1,2)
def test_two():
    return sp.Rational(1,3)
def test_three():
    return sp.Rational(1,5)
tests = [test_one(), test_two(), test_three()]
result = sum(tests) / len(tests)
print(result)