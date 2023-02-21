# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
import networkx as nx
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from various.network_tools import get_best_kr_equivalence, get_labels_from_Z

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = T
mode = "ALPHA"
topology = "MIX"
mapping="trivial"
index = "jacp"
# opt_score = ["_maxmu", "_X", "_D"]
opt_score = ["_maxmu"]

properties = {
  "version" : "ER",
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "mapping" : mapping,
  "topology" : topology,
  "index" : index,
  "mode" : mode
}

if __name__ == "__main__":
  rho = 0.2
  N = 128
  M = int(N * (N - 1) * rho)
  G = nx.gnm_random_graph(N, M, seed=12345, directed=T)
  A = nx.adjacency_matrix(G).todense()
  A = np.array(A, dtype=float)
  labels = np.arange(N).astype(int).astype(str)
  labels_dict = dict()
  for i in np.arange(N):
    labels_dict[i] = labels[i]
  # Create TOY ---
  NET = TOY(A, linkage, **properties)
  NET.set_alpha([6, 15, 30])
  NET.create_plot_directory()
  NET.set_labels(labels)
  H = Hierarchy(
    NET, A, A, np.zeros(A.shape),
    N, linkage, mode
  )
  ## Compute topologys ----
  H.BH_features_cpp()
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  # Set labels to network ----
  L = colregion(NET, labels=NET.labels)
  L.get_regions()
  H.set_colregion(L)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_X(on=T)
  plot_h.plot_measurements_mu(on=T)
  for j, score in enumerate(opt_score):
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    plot_h.lcmap_dendro(
      [k], cmap_name="husl",
      font_size=30, remove_labels=T,
      score="_"+score, on=T
    )
    plot_h.core_dendrogram(
      [r], score="_"+score,
      cmap_name="husl", remove_labels=T, on=T
    )
