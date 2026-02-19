import sympy as sp
import unittest

def parse_expr(expr_str):
    return sp.sympify(expr_str)

class ParserTest(unittest.TestCase):
    def test_simple_addition(self):
        self.assertEqual(parse_expr('2+3'), 5)
    def test_multiplication(self):
        self.assertEqual(parse_expr('4*5'), 20)
    def test_parentheses(self):
        self.assertEqual(parse_expr('2*(3+4)'), 14)

suite = unittest.TestLoader().loadTestsFromTestCase(ParserTest)
runner = unittest.TextTestRunner(verbosity=0)
result_obj = runner.run(suite)
result = result_obj.testsRun
print(result)