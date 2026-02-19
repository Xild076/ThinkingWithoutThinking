import sympy as sp

def shunting_yard(tokens):
    precedence = {'+':1, '-':1, '*':2, '/':2, '^':3}
    right_associative = {'^'}
    output = []
    stack = []
    for token in tokens:
        if token.replace('.','',1).isdigit() or (token.startswith('-') and token[1:].replace('.','',1).isdigit()):
            output.append(token)
        elif token.isalpha():
            output.append(token)
        elif token in precedence:
            while stack and stack[-1] in precedence and (
                (precedence[stack[-1]] > precedence[token]) or
                (precedence[stack[-1]] == precedence[token] and token not in right_associative)
            ):
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError('Mismatched parentheses')
            stack.pop()
        else:
            raise ValueError(f'Invalid token {token}')
    while stack:
        if stack[-1] in '()':
            raise ValueError('Mismatched parentheses')
        output.append(stack.pop())
    postfix = ' '.join(output)
    postfix = postfix.replace('^', '**')
    expr_sym = sp.sympify(postfix, locals={chr(i+97): sp.Symbol(chr(i+97)) for i in range(26)})
    return expr_sym

expr = '( 1 + 2 ) * 3'
tokens = expr.split()
result = sp.N(shunting_yard(tokens))
print(result)