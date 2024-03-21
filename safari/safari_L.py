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
import ctools as ct
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

    L_src = np.zeros(NET.nodes)
    L_tgt = np.zeros(NET.nodes)

    L_eff_src= np.zeros((NET.nodes, NET.nodes))
    L_eff_tgt= np.zeros((NET.nodes, NET.nodes))

    for i in np.arange(NET.nodes):
        psrc = NET.A[i, :]
        psrc = psrc / np.sum(psrc)
        # print(psrc)
        for l in np.arange(NET.nodes):
          if psrc[l] > 0:
             L_src[i] -= psrc[l] * np.log(psrc[l])
        ptgt = NET.A[:, i]
        # print(ptgt)
        for l in np.arange(NET.rows):
          if ptgt[l] > 0:
             L_tgt[i] -= ptgt[l] * np.log(ptgt[l])

    for i in np.arange(NET.nodes):
       for j in np.arange(i+1, NET.nodes):
          L_eff_src[i,j] = L_src[i] / 2 + L_src[j] / 2 + np.log(ct.Hellinger2(NET.A[i, :], NET.A[j, :], i, j))
          L_eff_tgt[i,j] = L_tgt[i] / 2 + L_tgt[j] / 2 + np.log(ct.Hellinger2(NET.A[:, i], NET.A[:, j], i, j))

          L_eff_src[j, i] = L_eff_src[i, j]
          L_eff_tgt[j, i] = L_eff_tgt[i, j]

    L_eff_src = adj2df(L_eff_src)
    L_eff_src = L_eff_src.loc[L_eff_src.source > L_eff_src.target]
    L_eff_tgt = adj2df(L_eff_tgt)
    L_eff_tgt = L_eff_tgt.loc[L_eff_tgt.source > L_eff_tgt.target]

    n = L_eff_src.shape[0]

    D = NET.D[:NET.nodes, :NET.nodes]
    D = adj2df(D)
    D = D.loc[D.source > D.target]

    data = pd.DataFrame(
       {
          "L" : list(L_eff_tgt.weight) + list(L_eff_src.weight),
          "set" : ["tgt"] * n + ["src"] * n,
          "distance" : list(D.weight) + list(D.weight)
       }
    )

    data = data.loc[data.L > -10]

    # sns.histplot(
    #    data=data,
    #    x="L", hue="set"
    # )
    

    sns.scatterplot(
       data=data,
      #  lowess=T,
       x="distance", y="L",
       hue="set",
       s=6
      #  scatter_kws={"s" : 6}
    )

    plt.show()