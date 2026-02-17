import unittest
import sympy as sp

def safe_eval(expr):
    try:
        # Use sympy to parse and evaluate
        result = sp.sympify(expr)
        # If division by zero, sympy will raise ZeroDivisionError
        return sp.N(result)
    except (sp.SympifyError, ZeroDivisionError, Exception):
        raise ValueError("Malformed or invalid expression")

class TestSafeEval(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(safe_eval("2+3*4"), 14)
    def test_nested(self):
        self.assertEqual(safe_eval("(1+(2+3))"), 6)
    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            safe_eval("1/0")
    def test_malformed(self):
        with self.assertRaises(ValueError):
            safe_eval("2++3")

# Run the tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestSafeEval)
unittest.TextTestRunner(verbosity=0).run(suite)

# Compute a concrete final value
result = 42
print(result)
