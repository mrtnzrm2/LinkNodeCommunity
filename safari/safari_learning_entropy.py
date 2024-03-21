import numpy as np

x = np.random.randint(100, size=(4,))
px = np.random.rand(4)
x[px <= 0.5] = 0
x = x / np.sum(x)

y = np.random.randint(100, size=(4,))
py = np.random.rand(4)
y[py <= 0.5] = 0
y = y / np.sum(y)

messx = np.zeros(4)


for i in np.arange(10000):
      ij = np.random.multinomial(1, pvals=x)
      messx[np.argmax(ij)] += 1

messy = np.zeros(4)

for i in np.arange(10000):
      ij = np.random.multinomial(1, pvals=y)
      messy[np.argmax(ij)] += 1

print(x, y)
print(messx, messy)

minmess = np.sum([np.minimum(i, j) for i, j in zip(messx, messy)])
maxmess = np.sum([np.maximum(i, j) for i, j in zip(messx, messy)])

print(- np.log(minmess / maxmess))
print(-2 * np.log(np.sum([np.sqrt(i * j) for i, j in zip(x, y)])))







