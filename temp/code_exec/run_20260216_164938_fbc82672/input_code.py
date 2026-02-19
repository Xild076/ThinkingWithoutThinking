from io import StringIO
import contextlib
buf = StringIO()
with contextlib.redirect_stdout(buf):
    for i in range(1, 101):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
output = buf.getvalue()
result = output
print(result)