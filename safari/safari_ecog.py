# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False

# Starndard libraries ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from networks.ECoG.structure import WAVES
from modules.hierarmerge import Hierarchy
from modules.colregion import colECoG
from various.network_tools import adj2df

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
mode = "ZERO"
nature = "MK1PreGamma"
topology = "MIX"
mapping = "trivial"
index  = "D1_2_3"
opt_score = ["_SD", "_X"]

NET = WAVES[nature](
    linkage=linkage,
    mode=mode,
    nlog10=nlog10,
    lookup=lookup,
    cut=cut,
    topology=topology,
    mapping=mapping,
    index=index
)

# NET.C[NET.C < 0.01] = 0
E = np.sum(NET.C > 0)
rho = E / (NET.nodes * (NET.nodes - 1))

print(E, rho)
print(np.max(NET.C))
print(np.min(NET.C[NET.C > 0]))
C = NET.C.copy()
C[C > 0] = np.log(C[C > 0])
C = (C - C.T) / 2
C = adj2df(C)
C = C.loc[C.source > C.target]


sns.histplot(
    data=C,
    x="weight"
)

plt.show()
# H = Hierarchy(NET, NET.C, NET.C, NET.D, NET.nodes, linkage, mode, lookup=lookup)

# yup = NET.C != 0
# data = {
#     "GC" : np.log10(NET.C[yup].ravel()),
#     "dist" : NET.D[yup].ravel()
# }

# _, ax = plt.subplots(1, 2)
# sns.histplot(
#     data=data,
#     x="GC",
#     ax=ax[0]
# )
# sns.scatterplot(
#     data=data,
#     x="dist",
#     y="GC",
#     s=1,
#     ax=ax[1]
# )
# plt.show()


