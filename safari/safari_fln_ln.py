# Insert path ---
import os
import sys
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from various.data_transformations import maps
from networks.structure import MAC
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "LN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
bias = 0.
alpha = 0.
opt_score = ["_S", "_SD", "_X"]
save_data = F
version = "57d106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = MAC[f"MAC{__inj__}"](
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
      b = bias,
      alpha = alpha
    )

    H = Hierarchy(
      NET, NET.C, NET.C, NET.D,
      __nodes__, linkage, mode, lookup=lookup
    )

    Dsource = H.source_sim_matrix
    Dsource[Dsource == 0] = np.nan
    Dsource = 1/Dsource + 1
    Dsource[np.isnan(Dsource)] = np.nanmax(Dsource) + np.nanstd(Dsource)
    np.fill_diagonal(Dsource, np.nan)
    Dtarget = H.target_sim_matrix
    Dtarget[Dtarget == 0] = np.nan
    Dtarget = 1/Dtarget + 1
    Dtarget[np.isnan(Dtarget)] = np.nanmax(Dtarget) + np.nanstd(Dtarget)
    np.fill_diagonal(Dtarget, np.nan)

    Dsource = adj2df(Dsource)
    Dsource = Dsource.loc[(Dsource.source > Dsource.target) & (~np.isnan(Dsource.weight))]
    Dsource["dir"] = "source"
    Dtarget = adj2df(Dtarget)
    Dtarget["dir"] = "target"
    Dtarget = Dtarget.loc[(Dtarget.source > Dtarget.target) & (~np.isnan(Dtarget.weight))]
    D = pd.concat([Dsource, Dtarget], ignore_index=True)
    
    d2 = np.exp(np.mean(np.log(D.weight)))
    var2 = np.exp(np.std(np.log(D.weight)))
    skew2 = np.exp(skew(np.log(D.weight)))
    sns.histplot(
        data=D,
        x="weight",
        hue="dir"
    )
    plt.axvline(d2)
    plt.axvline(d2 + var2 / 2, color="orange")
    plt.axvline(d2 + skew2 / 3, color="green")
    plt.xscale("log")
    plt.show()
  