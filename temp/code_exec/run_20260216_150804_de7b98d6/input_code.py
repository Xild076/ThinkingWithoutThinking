messages = ['hello', 'POISON', 'world', 'another POISON', 'safe']
poison_count = sum(1 for msg in messages if 'POISON' in msg)
result = poison_count
print(result)