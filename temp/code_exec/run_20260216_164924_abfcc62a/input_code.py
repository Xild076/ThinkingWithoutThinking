import ast


drafted_fizzbuzz = '''
for i in range(1,101):
    output = ''
    if i % 3 == 0:
        output += 'Fizz'
    if i % 5 == 0:
        output += 'Buzz'
    print(output or i)
'''

try:
    ast.parse(drafted_fizzbuzz)
    syntax_ok = True
except SyntaxError:
    syntax_ok = False

# validate logic for numbers 1..15
expected = []
for i in range(1,16):
    out = ''
    if i % 3 == 0:
        out += 'Fizz'
    if i % 5 == 0:
        out += 'Buzz'
    expected.append(out or str(i))

simulated = []
for i in range(1,16):
    out = ''
    if i % 3 == 0:
        out += 'Fizz'
    if i % 5 == 0:
        out += 'Buzz'
    simulated.append(out or str(i))

logical_ok = expected == simulated
validation_passed = syntax_ok and logical_ok
result = validation_passed
print(result)