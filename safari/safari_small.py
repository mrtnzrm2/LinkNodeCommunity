# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
import seaborn as sns
sns.set_theme()
# Personal libs ----
from networks.structure import MAC
from various.network_tools import column_normalize


# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
version = 220830
mode = "ALPHA"
nature = "original"
distance = "MAP3D"
topology = "MIX_JACW"
mapping = "R2"
index="jacw"
imputation_method = ""
opt_score = ["_maxmu"]
save_data = T
__nodes__ = 57
__inj__ = 57

properties = {
  "version" : version,
  "prob" : prob,
  "distance": distance,
  "cut" : cut,
  "nature" : nature,
  "inj" : __inj__,
  "sln" : F
}

if __name__ == "__main__":
  mac = MAC(
    linkage, mode, nlog10=nlog10,
    lookup=lookup, topology=topology,
    mapping=mapping, index=index,
    not_path=T, **properties
  )
  mac.C[mac.C == 0] = 1
  np.fill_diagonal(mac.C, 0)
  mac.C = column_normalize(mac.C)
  mac.C[mac.C != 0] = -np.log(mac.C[mac.C != 0])
  b = np.max(mac.C[mac.C != 0])
  print(b)

