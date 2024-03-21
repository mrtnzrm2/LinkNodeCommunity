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
from modules.hierarentropy import Hierarchical_Entropy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.discovery import discovery_channel
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
mode = "ZERO"
topology = "MIX"
mapping="trivial"
index = "Hellinger2"
# opt_score = ["_maxmu", "_X", "_D"]
opt_score = ["_S"]

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
  rho = 0.6
  N = 30
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
  NET.create_plot_directory()
  NET.set_labels(labels)
  H = Hierarchy(
    NET, A, A, np.zeros(A.shape),
    N, linkage, mode
  )
  ## Compute features ----
  H.BH_features_cpp_no_mu()
  ## Compute link entropy ----
  H.link_entropy_cpp("short", cut=cut)
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  ## Compute node entropy ----
  H.node_entropy_cpp("short", cut=cut)
  ## Update entropy ----
  H.entropy = [
    H.node_entropy, H.node_entropy_H,
    H.link_entropy, H.link_entropy_H
  ]
  # Set labels to network ----
  L = colregion(NET, labels=NET.labels)
  L.get_regions()
  H.set_colregion(L)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_n = Plot_N(NET, H)
  # HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  # HS.Z2dict("short")
  # HS.zdict2newick(HS.tree, weighted=F, on=T)
  # plot_h.plot_newick_R(
  #   HS.newick, HS.total_nodes,
  #   threshold=H.node_entropy[0].shape[0] - np.argmax(H.node_entropy[0]) - 1,
  #   weighted=F, on=T
  # )
  # HS.zdict2newick(HS.tree, weighted=T, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=T, on=T)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_S(on=T)
  for j, score in enumerate(opt_score):
    K, R, HT = get_best_kr_equivalence(score, H)
    for k, r in zip(K, R):
      rlabels = get_labels_from_Z(H.Z, r)
      rlabels = skim_partition(rlabels)
      NET.overlap, NET.data_nocs, sizes, rlabels2 = discovery_channel["discovery_7"](H, k, rlabels, index=index, direction="both",  undirected=F)
      H.set_overlap_labels(NET.overlap, score, "both")
      # plot_h.lcmap_dendro(
      #   [k], cmap_name="hsl",
      #   font_size=30, remove_labels=T,
      #   score="_"+score, on=T
      # )
    # plot_h.core_dendrogram(
    #   [r], score="_"+score,
    #   cmap_name="husl", remove_labels=T, on=T
    # )
      plot_n.plot_network_covers(
        k, NET.A, rlabels2,
        NET.data_nocs, sizes, H.colregion.labels[:H.nodes],
        score=score, direction="both", cmap_name="hls", on=T, figsize=(8,8)
      )
