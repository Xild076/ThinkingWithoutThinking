class RollingWindowStats:
    def __init__(self, size):
        self.size = size
        self.values = []
        self.total = 0
        self.total_sq = 0
    def add(self, x):
        self.values.append(x)
        self.total += x
        self.total_sq += x*x
        if len(self.values) > self.size:
            old = self.values.pop(0)
            self.total -= old
            self.total_sq -= old*old
    def stats(self):
        return self.total, self.total_sq, len(self.values)

# Example usage
entity = RollingWindowStats(3)
for v in [1, 2, 3, 4]:
    entity.add(v)
result = entity.total_sq
print(result)
