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
import ctools as ct
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

    nbe = NET.nodes * (NET.nodes - 1) // 2
    nbre = nbe * (nbe - 1) // 2

    sm = np.zeros(nbe)
    tm = np.zeros(nbe)

    A = NET.A

    e = 0
    for i in np.arange(NET.nodes):
        for j in np.arange(i+1, NET.nodes):
            sm[e] = -2 * np.log(ct.Hellinger2(A[i, :], A[j, :], i, j))
            tm[e] = -2 * np.log(ct.Hellinger2(A[:, i], A[:, j], i, j))
            e += 1


    rel_sm = np.zeros(nbre)
    rel_tm = np.zeros(nbre)

    e = 0
    for i in np.arange(nbe):
        for j in np.arange(i+1, nbe):
            rel_sm[e] = np.abs(sm[i] - sm[j])
            rel_tm[e] = np.abs(tm[i] - tm[j])
            e += 1


    data = pd.DataFrame(
        {
            "Relative information" : list(rel_sm) + list(rel_tm),
            "set" : ["source"] * nbre + ["target"] * nbre
        }
    )

    sns.histplot(
        data=data,
        x="Relative information",
        hue="set",
        stat="density"
    )

    plt.show()