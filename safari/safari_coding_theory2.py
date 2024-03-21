# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject = "MAC"
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
version = "57"+"d"+"106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = STR[f"{subject}{__inj__}"](
      linkage, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      discovery = discovery,
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = alpha
    )

    i1 = match(["v2c"], NET.struct_labels)[0]
    i2 = match(["v1c"], NET.struct_labels)[0]

    U1 = NET.A[:, i1]
    U2 = NET.A[:, i2]

    s = int(1e6)

    m=1
    # Pa = np.random.rand(s * m)

    L1 = np.random.choice(np.arange(NET.rows), m*s, p=U1)
    L2 = np.random.choice(np.arange(NET.rows), m*s, p=U2)

    L1F = np.zeros(s)
    L2F = np.zeros(s)

    A = np.zeros((NET.rows, NET.rows))

    for i in np.arange(m * s):
       A[L1[i], L2[i]] += 1

    A /= (m * s)

    P1 = np.sum(A, axis=1).ravel()
    P2 = np.sum(A, axis=0).ravel()

    Iuv = 0

    for i in np.arange(NET.nodes):
       for j in np.arange(NET.nodes):
          if A[i,j] == 0 and P1[i] != 0 and P2[j] != 0: continue
          Iuv += A[i,j] * np.log(A[i,j] / (P1[i]*P2[j]))

    AA = A.copy()
    AA[AA == 0] =np.nan
    sns.heatmap(
       np.log(AA),
       xticklabels=NET.struct_labels,
       yticklabels=NET.struct_labels
    )

    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=5)
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=5)
    plt.show()

    print(Iuv)

    # L1 = np.zeros(s*m).astype(int)
    # L2 = np.zeros(s*m).astype(int)

    # for i in np.arange(s * m):
    #     if U1[Draws2[i]] > Pa[i]:
    #         L1[i] = 1

    #     if U2[Draws2[i]] > Pa[i]:
    #         L2[i] = 1

    # L1F = np.zeros(s)
    # L2F = np.zeros(s)

    for i in np.arange(s):
      L1F[i] = np.round(np.mean(L1[(m * i):(m*(i+1))]))
      L2F[i] = np.round(np.mean(L2[(m * i):(m*(i+1))]))

    L1F = L1F.astype(int)
    L2F = L2F.astype(int)

    print(np.unique(L1F).shape)
    print(np.unique(L2F).shape)

    print(L1F[:40])
    print(L2F[:40])

    from sklearn.metrics import mutual_info_score 
    Iuv = mutual_info_score(L1F, L2F)

    print(Iuv)

