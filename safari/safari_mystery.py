import numpy as np


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


for i in np.linspace(-12, -1, 100):
  a = np.array([1, 12])
  b = np.array([1, 12-np.exp(i)])
  print(i, jacp_smooth(a, b, 2))