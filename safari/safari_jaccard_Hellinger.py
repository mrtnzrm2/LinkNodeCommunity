# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from various.network_tools import adj2df

def jaccard(u, v) :
  return np.dot(u, v) / (np.power(np.linalg.norm(u), 2) + np.power(np.linalg.norm(v), 2) - np.dot(u, v))

def H2(u, v) :
  return np.sum(np.power(np.sqrt(u) - np.sqrt(v), 2)) / 2

def R12(u, v):
  return -2 * np.log(np.sum(np.sqrt(np.dot(u, v))))

M = 100
N = 25

a = np.random.randint(2, size=(M, N)) * 1.
pa = a.T / np.sum(a, axis=1)
pa = pa.T

J = np.zeros((M, M))
H = np.zeros((M, M))

for i in np.arange(M):
  for j in np.arange(i + 1, M):
    J[i, j] = jaccard(a[i, :], a[j, :])
    J[j, i] = J[i, j]
    H[i, j] = R12(pa[i, :], pa[j, :])
    H[j, i] = H[i, j]

J = adj2df(J)
J = J.loc[J.source > J.target]

H = adj2df(H)
H = H.loc[H.source > H.target]

data = {
  "J" : 1 - J["weight"].ravel(),
  "H" : H["weight"].ravel()
}

sns.scatterplot(
  data=data,
  x="H",
  y="J"
)

plt.show()

