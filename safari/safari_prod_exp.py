# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
##
import numpy as np
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
from various.network_tools import adj2df

def simple(x, y):
  return -np.mean(np.abs(x-y))

def simple2(x, y):
  return np.mean(x+y)

n = 100
x = np.random.uniform(size=(10000, n))
x = - np.log(x) / 0.079
x += 1
x = np.log(x)
y = np.zeros((n, n))
for i in np.arange(n):
  for j in np.arange(i, n):
    y[i, j] = simple(x[:, i], x[:, j])
    y[j, i] = y[i, j]

y = adj2df(y)
y = y.loc[y.source > y.target]["weight"].to_numpy()

data = pd.DataFrame({"x" : y.ravel()})
_, ax = plt.subplots(1,1)
sns.histplot(data=data, x="x", stat="density", ax=ax)
plt.show()