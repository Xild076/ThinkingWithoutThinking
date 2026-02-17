expected = ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
output = []
for i in range(1,16):
    if i%3==0 and i%5==0:
        output.append("FizzBuzz")
    elif i%3==0:
        output.append("Fizz")
    elif i%5==0:
        output.append("Buzz")
    else:
        output.append(str(i))
description = "FizzBuzz is a classic programming challenge where numbers are replaced by 'Fizz', 'Buzz', or 'FizzBuzz' based on divisibility."
result = (output == expected and len(description.split()) <= 150)
print(result)