import sympy as sp
import unittest

def replay_engine(seed):
    x = sp.Symbol('x')
    expr = sp.sin(x) + sp.cos(x)
    return float(expr.subs(x, seed))

class TestReplayEngineDeterminism(unittest.TestCase):
    def test_deterministic_output(self):
        for i in range(5):
            self.assertAlmostEqual(replay_engine(i), replay_engine(i))
    def test_multiple_scenarios(self):
        outputs = [replay_engine(i) for i in range(10)]
        self.assertEqual(len(outputs), 10)

# Run tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestReplayEngineDeterminism)
unittest.TextTestRunner(verbosity=0).run(suite)

# Compute final value
result = sum([replay_engine(i) for i in range(10)])
print(result)
