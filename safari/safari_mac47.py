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
import networkx as nx
# Personal libs ---- 
from networks.MAC.mac47 import MAC47
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.network_tools import *

def plot_network_raw(R, labels, path):
  from pathlib import Path
  import matplotlib.patheffects as path_effects
  edges = np.sum(R >0)
  # Generate graph ----
  G = nx.DiGraph(R)
  r_min = np.min(R[R>0])
  r_max = np.max(R[R < np.Inf])
  edge_color = ["#666666"] * edges
  pos = nx.kamada_kawai_layout(G, weight="kk_weight")
  for i, dat in enumerate(G.edges(data=True)):
      u, v, a = dat
      G[u][v]["sp_weight"] = - (a["weight"] - r_min) + r_max
  # pos = nx.spring_layout(G, weight="sp_weight", pos=pos, iterations=5, seed=212)

  plt.style.use("dark_background")
  sns.set_context("talk")
  plt.figure(figsize=(12, 12))

  nx.draw_networkx_edges(
    G, pos=pos, edge_color=edge_color, alpha=0.5, width=2, arrowsize=15, connectionstyle="arc3,rad=-0.1",
    node_size=1400, arrowstyle="->"
  )

  labs = {k : lab for k, lab in zip(G.nodes, labels)}
  t = nx.draw_networkx_labels(
     G, pos=pos, labels=labs, font_size=24, font_color="white", font_family="Times New Roman"
  )
  for key in t.keys():
    t[key].set_path_effects(
    [
      path_effects.Stroke(linewidth=1, foreground='black'),
      path_effects.Normal()
    ]
  )
    
  nx.draw_networkx_nodes(
     G, pos=pos, node_color="#FC6255", edgecolors="white", node_size=1400,
     linewidths=0.5
  )

  array_pos = np.array([list(pos[v]) for v in pos.keys()])
  plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
  plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
  plt.grid(False)
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  
  plot_path = os.path.join(
    path, "Network"
  )
  print(plot_path)
  # Crate path ----
  Path(
    plot_path
  ).mkdir(exist_ok=True, parents=True)
  # Save plot ----

  #### Careful: manual modification #### -----
  plt.savefig(
    os.path.join(
      plot_path, f"net_raw_dark.svg"
    ),
    dpi=300, transparent=True
  )
  plt.close()

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
bias = 0
opt_score = ["_S"]
save_data = T
version = "47d106"
__nodes__ = 47
__inj__ = 47
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC47(
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


  R = NET.A[:__nodes__, :].copy()
  R[R > 0] = -np.log(R[R > 0])

  plot_network_raw(R, NET.struct_labels[:NET.nodes], NET.plot_path)
