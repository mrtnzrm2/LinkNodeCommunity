# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
# Personal libs ---- 
from networks.MAC.mac29 import MAC29
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
bias = 0
opt_score = ["_S", "_X", "_SD"]
save_data = T
version = "29d91"
__nodes__ = 29
__inj__ = 29
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC29(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )