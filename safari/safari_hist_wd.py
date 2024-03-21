# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps
from networks.structure import STR
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
subject = "MAC"
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
# imputation_method = "RF2"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "discovery_7"
opt_score = ["_S"]
save_data = F
__nodes__ = 57
__inj__ = f"{__nodes__}"
version = f"{__nodes__}"+"d"+"106"

# Load structure ----
NET = STR[f"{subject}{__inj__}"](
  linkage, mode,
  nlog10 = nlog10,
  structure = structure,
  lookup = lookup,
  version = version,
  nature = nature,
  distance = distance,
  inj = __inj__,
  topology = topology,
  index = index,
  mapping = mapping,
  cut = cut,
  b = bias,
  alpha = alpha,
  discovery = discovery
)

W = NET.A
W[W!=0] = np.log10(W[W!=0]) + 7

D = NET.D

W = adj2df(W)
W = W.loc[W["weight"] != 0]
W["weight"] = W["weight"] / np.max(W["weight"])

D = adj2df(D)
D = D.loc[D["source"] > D["target"]]
D["weight"] = D["weight"] / np.max(D["weight"])



data = {
  "val" : list(D["weight"]) +  list(W["weight"]),
  "var" : ["dist"] * D.shape[0] + ["weight"] * W.shape[0]
}


import seaborn as sns
import matplotlib.pyplot as plt

# xbins = np.power(10, np.linspace(-7, 0, 10))

sns.histplot(
  data=data,
  x="val",
  stat="density",
  common_bins=False,
  alpha=0.3,
  common_norm=False,
  # cumulative=T,
  # multiple="dodge",

  hue="var"
)

plt.savefig("../plots/two_histograms.png", dpi=300)

