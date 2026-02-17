import unittest

def tokenize(expr):
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or c == '.':
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(('NUMBER', expr[i:j]))
            i = j
        elif c in '+-*/()':
            if c == '+':
                tokens.append(('PLUS', c))
            elif c == '-':
                tokens.append(('MINUS', c))
            elif c == '*':
                tokens.append(('TIMES', c))
            elif c == '/':
                tokens.append(('DIVIDE', c))
            elif c == '(':
                tokens.append(('LPAREN', c))
            elif c == ')':
                tokens.append(('RPAREN', c))
            i += 1
        else:
            raise SyntaxError(f'Unexpected character {c} at {i}')
    tokens.append(('EOF', ''))
    return tokens

class PrattParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
    def peek(self):
        return self.tokens[self.pos][0]
    def consume(self, expected_type):
        typ, val = self.tokens[self.pos]
        if typ != expected_type:
            raise SyntaxError(f'Expected {expected_type} but got {typ}')
        self.pos += 1
        return val
    def parse_expression(self, precedence):
        token_type, value = self.tokens[self.pos]
        if token_type == 'NUMBER':
            left = float(value)
            self.consume('NUMBER')
        elif token_type == 'LPAREN':
            self.consume('LPAREN')
            left = self.parse_expression(0)
            self.consume('RPAREN')
        else:
            raise SyntaxError('Expected number or "("')
        while self.peek() in ('PLUS', 'MINUS', 'TIMES', 'DIVIDE'):
            op_type = self.peek()
            if precedence >= PRECEDENCE[op_type]:
                break
            self.consume(op_type)
            right_prec = PRECEDENCE[op_type] + 1
            right = self.parse_expression(right_prec)
            if op_type == 'PLUS':
                left = left + right
            elif op_type == 'MINUS':
                left = left - right
            elif op_type == 'TIMES':
                left = left * right
            elif op_type == 'DIVIDE':
                left = left / right
        return left
    def parse(self):
        return self.parse_expression(0)

PRECEDENCE = {
    'TIMES': 2,
    'DIVIDE': 2,
    'PLUS': 1,
    'MINUS': 1,
}

def evaluate(expr):
    tokens = tokenize(expr)
    parser = PrattParser(tokens)
    return parser.parse()

class TestPrattParser(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(evaluate('2+3'), 5)
    def test_precedence(self):
        self.assertEqual(evaluate('2+3*4'), 14)
    def test_parentheses(self):
        self.assertEqual(evaluate('(2+3)*4'), 20)
    def test_negative(self):
        self.assertEqual(evaluate('-2'), -2)
    def test_division(self):
        self.assertAlmostEqual(evaluate('8/3'), 8/3)

TestPrattParser().test_simple()
TestPrattParser().test_precedence()
TestPrattParser().test_parentheses()
TestPrattParser().test_negative()
TestPrattParser().test_division()
result = evaluate('2+3*4')
print(result)
