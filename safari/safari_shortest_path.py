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
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
# Personal libs ---- 
from networks.MAC.mac40 import MAC40
from modules.discovery import discovery_channel
from various.data_transformations import maps
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "40d91"
__nodes__ = 40
__inj__ = 40
  
# Start main ----9
if __name__ == "__main__":
  # Load structure ----
  NET = MAC40(
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

  H = Hierarchy(
    NET, NET.A, NET.A, NET.D,
    __nodes__, linkage, mode, lookup=lookup,
    index=index
  )

  tgt = -2 * np.log(H.target_sim_matrix)

  B = NET.A[:NET.nodes, :NET.nodes]
  B = -np.log(B)

  import networkx as nx

  GB = nx.DiGraph(B)

  sw  = []
  TGT = []
  d = []
  for i in np.arange(NET.nodes):
    for j in np.arange(NET.nodes):
      if i == j: continue
      d.append(NET.D[i,j])
      TGT.append(tgt[i,j])
      sw.append(nx.shortest_path_length(GB, source=i, target=j, weight="weight"))


  sns.set_style("ticks")

  fig, ax = plt.subplots(1,1)
  ax.minorticks_on()

  sns.scatterplot(
    x=TGT, y=sw, ax=ax, s=10, alpha=0.6
  )
  ax.set_xlabel("D1/2")
  ax.set_ylabel("Shortest path distance")
  sns.despine(ax=ax, top=T, right=T)

  plot_path = NET.plot_path

  from pathlib import Path
  Path(
    plot_path
  ).mkdir(exist_ok=True, parents=True)


  # sns.histplot(x=sw, ax=ax[1])
  # ax[1].set_xlabel("Shortest path distance")

  # fig.set_figheight(4)
  # fig.set_figwidth(8)
  fig.tight_layout()

  # Save plot ----
  plt.savefig(
    os.path.join(
      plot_path, f"Features/shortet_d12_40d40.png"
    ),
    dpi=300
  )


