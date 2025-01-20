
l = 0
u = 1
a = 6
b = 4

mu = l + (u - l) * (a / (a + b))
print(mu)
u = (mu * (a + b)) / a
print(u)

l += .1
u = ((mu - l) * (a + b)) / a
print(u)
mu = l + (u - l) * (a / (a + b))
print(mu)

print(17141/2)