import sympy as sp
import unittest
import io
import sys

def parse_expr(expr_str):
    return sp.sympify(expr_str)

class ParserTest(unittest.TestCase):
    def test_simple_addition(self):
        self.assertEqual(parse_expr('2+3'), 5)
    def test_multiplication(self):
        self.assertEqual(parse_expr('4*5'), 20)
    def test_parentheses(self):
        self.assertEqual(parse_expr('2*(3+4)'), 14)

stream = io.StringIO()
sys_stdout = sys.stdout
sys.stdout = stream
suite = unittest.TestLoader().loadTestsFromTestCase(ParserTest)
runner = unittest.TextTestRunner(stream=stream, verbosity=2)
result_obj = runner.run(suite)
sys.stdout = sys_stdout
final_value = result_obj.testsRun
result = final_value
print(result)