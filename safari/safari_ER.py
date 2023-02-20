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
from plotting_modules.plotting_overlap import PLOT_O
from various.network_tools import get_best_kr_equivalence, get_labels_from_Z

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = T
mode = "ALPHA"
distance = ""
topology = "MIX"
mapping="trivial"
opt_score = ["_maxmu", "_X", "_D"]
# opt_score = ["_maxmu"]

properties = {
  "version" : "ER",
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "distance": distance,
  "cut" : cut,
  "topology" : topology,
  "mode" : mode
}

if __name__ == "__main__":
  rho = 0.6
  N = 128
  L = int(N * (N - 1) * 0.57)
  G = nx.gnm_random_graph(N, L, seed=12345, directed=T)
  A = nx.adjacency_matrix(G).todense()
  A = np.array(A, dtype=float)
  labels = np.arange(N).astype(int).astype(str)
  labels_dict = dict()
  for i in np.arange(N):
    labels_dict[i] = labels[i]
  # Create TOY ---
  NET = TOY(A, linkage, **properties)
  NET.set_labels(labels)
  H = Hierarchy(
    NET, A, np.zeros(A.shape),
    N, linkage, mode, prob=prob
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
  plot_h.Mu_plotly(on=T) #
  plot_h.D_plotly(on=T) #
  plot_h.X_plotly(on=T) #
  plot_h.order_parameter_plotly(on=T) #
  plot_h.susceptibility_plotly(on=T) #
  # Plot O ----
  plot_o = PLOT_O(NET, H)
  for j, score in enumerate(opt_score):
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    plot_h.lcmap_dendro(
      [k], cmap_name="husl",
      font_size=30,
      score="_"+score, on=T
    )
    plot_h.plot_networx(
      r, rlabels, score="_"+score,
      on=T, labels=labels_dict, cmap_name="husl"
    )
    plot_h.plot_networx_link_communities(
      [k], score="_"+score,
      cmap_name="husl",
      on=T, labels=labels_dict
    )
    plot_h.core_dendrogram(
      [r], score="_"+score,
      on=T, cmap_name="husl"
    )
    plot_o.bar_node_membership(
      [k], labels = rlabels, score="_"+score,
      node_labels = labels, on=T,
    )
    plot_o.bar_node_overlap(
      [k], NET.overlap, score="_"+score,
      node_labels = labels, on=T
    )
