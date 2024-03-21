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
from pathlib import Path
# Personal libraries ----
from networks.structure import MAC
from various.network_tools import *

def weight_histograms(NET):
    C = NET.C.copy()
    C = np.log(1 + C)
    labels = NET.struct_labels[:__nodes__]
    C = C[:,:__nodes__][:__nodes__,:]
    C = adj2df(C)
    C = C.loc[C.weight > 0]
    C["SOURCE"] = labels[C.source]
    C["TARGET"] = labels[C.target]
    
    data = pd.DataFrame()
    focus = ["8l", "v1c", "1", "f1", "5", "v1fpuf",  "v1pclf"]
    mean = []
    mean_x = []
    for f in focus:
        sub = C.loc[(C.SOURCE == f) | (C.TARGET == f)]
        mean.append(np.mean(sub.weight))
        sub["focus"] = f
        data = pd.concat([data, sub], ignore_index=T)

    g=sns.FacetGrid(
        data=data,
        col="focus",
        col_wrap=4,
        sharex=F
    )
    g.map_dataframe(
        sns.histplot,
        x="weight",
        kde=T
    )
    for i, ax in enumerate(g.axes.flat):
        ax.axvline(mean[i], ls="--")
    plt.show()

def binary_histograms(NET):
    C = NET.C.copy()
    C = np.log(1 + C)
    labels = NET.struct_labels[:__nodes__]
    C = C[:,:__nodes__][:__nodes__,:]
    C = adj2df(C)
    C.weight.loc[C.weight > 0] = 1
    C["SOURCE"] = labels[C.source]
    C["TARGET"] = labels[C.target]
    
    data = pd.DataFrame()
    focus = ["8l", "v1c", "1", "f1", "5", "v1fpuf", "v1pclf"]
    mean = []
    for f in focus:
        sub = C.loc[(C.SOURCE == f) | (C.TARGET == f)]
        mean.append(np.mean(sub.weight))
        sub["focus"] = f
        data = pd.concat([data, sub], ignore_index=T)

    g=sns.FacetGrid(
        data=data,
        col="focus",
        col_wrap=4,
        sharex=F
    )
    g.map_dataframe(
        sns.histplot,
        x="weight",
        kde=T
    )
    for i, ax in enumerate(g.axes.flat):
        ax.axvline(mean[i], ls="--")
    plt.show()

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
opt_score = ["_X", "_S"]
save_data = T
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
      alpha = alpha,
      discovery="discovery_9"
    )
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )
    
    src = 1 - H.source_sim_matrix
    tgt = 1 - H.target_sim_matrix

    xsrc, ysrc = np.where(src < 0.33)
    xtgt, ytgt = np.where(tgt < 0.33)

    SRC = np.zeros(src.shape)
    TGT = np.zeros(tgt.shape)

    SRC[xsrc, ysrc] = 1
    TGT[xtgt, ytgt] = 2

    G = SRC + TGT
    G[G == 0] = np.nan

    sns.heatmap(
        data=G,
        xticklabels=NET.struct_labels[:57],
        yticklabels=NET.struct_labels[:57],
        cmap=sns.color_palette("deep", 3),
        cbar=False
    )