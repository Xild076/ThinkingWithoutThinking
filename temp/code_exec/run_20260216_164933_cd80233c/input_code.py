import sys
from io import StringIO
old_stdout = sys.stdout
sys.stdout = captured = StringIO()
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
sys.stdout = old_stdout
output = captured.getvalue()
result = output
print(result)