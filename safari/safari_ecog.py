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
from modules.colregion import colECoG

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
mode = "ALPHA"
nature = "MK2PostBeta"
topology = "MIX"
mapping = "trivial"
index  = "D1_2_2"
opt_score = ["_maxmu", "_X"]
save_data = T

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

NET.C[NET.C < 0.01] = 0

E = np.sum(NET.C > 0)
rho = E / (NET.nodes * (NET.nodes - 1))

print(E, rho)

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


