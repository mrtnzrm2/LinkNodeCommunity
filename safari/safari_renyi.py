import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta

def D1_2_beta(a, b, x, y):
  return np.log(beta(a, b) * beta(x, y) / np.power(beta((a + x) / 2, (b + y)/2), 2))


one = np.linspace(0.1, 2, 20)

x, y = np.meshgrid(one, one)

z = np.zeros(x.shape)

for i in np.arange(20):
  for j in np.arange(20):
    z[i, j] = D1_2_beta(x[i, j], 2, y[i, j], 2)

print(z)

plt.imshow(z)

plt.show()