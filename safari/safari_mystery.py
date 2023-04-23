import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def jacp_smooth(u, v, n : int, *args):
  if n > 0:
    p = 0
    for i in np.arange(n):
      q = np.sum(np.log(1 + np.maximum(np.exp(u - u[i]), np.exp(v - v[i]))))
      p += np.log(2) / q
    return p
  else: return np.nan

def jacp(u, v, n : int, *args):
  if n > 0:
    U = np.tile(u, (n, 1))
    U = U / u[:, None]
    U[np.isnan(U)] = np.Inf
    V = np.tile(v, (n, 1))
    V = V / v[:, None]
    V[np.isnan(V)] = np.Inf
    return np.sum(1 / np.sum(np.maximum(U, V), axis=1))
  else: return np.nan

x = np.linspace(-12, 0, 1000)
jp = np.zeros(1000)
for e, i in enumerate(x):
  a = np.array([1, 2])
  b = np.array([1, 2-np.exp(i)])
  jp[e] = jacp_smooth(a, b, 2)

data = pd.DataFrame(
  {
    "x" : np.exp(x),
    "jacp_smooth" : jp 
  }
)

print(np.max(jp))
print(np.min(jp))
_, ax = plt.subplots(1, 1)
sns.lineplot(
  data=data,
  x="x",
  y="jacp_smooth",
  ax=ax
)

ax.set_xscale("log")

plt.show()