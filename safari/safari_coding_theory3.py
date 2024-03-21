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

    T = int(1e3)
    nodes = NET.nodes
    labels = NET.struct_labels

    F = np.zeros((nodes, T))

    seeds = ["v1c", "v1fplf"]
    seeds = match(seeds, labels)

    F[seeds, :5] = 1

    Px_y = NET.A

    length_tr = int(1e6)
    Ptr = np.random.rand(length_tr)

    def recharge_Ptr():
        return np.random.rand(length_tr)

    index_tr = 0
    for t in np.arange(1, T):
        for x in np.arange(nodes):
            if F[x, t-1] > 0:
              for y in np.arange(nodes):
                  if x == t: continue
                  
                  if t < T-3:
                    if Px_y[x, y] > Ptr[index_tr]:
                        F[y, t:(t+3)] += 1
                        if t < T-4:
                            F[y, (t+3):(t+4)] -= 1


                  index_tr +=1

                  if index_tr == length_tr:
                      Ptr = recharge_Ptr()
                      index_tr = 0
                  

    fig, ax = plt.subplots(1, 1)

    sns.heatmap(
        F,
        yticklabels=labels[:nodes],
        ax=ax
    )

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

    plt.show()