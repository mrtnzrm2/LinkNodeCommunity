# Insert path ---
import os
import sys
import seaborn as sns
sns.set_theme()
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from modules.hierarentropy import Hierarchical_Entropy
from networks.toy import TOY
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.discovery import discovery_channel
from various.network_tools import *

G = nx.karate_club_graph()

nodes = 34
A = np.zeros((34, 34))
for u, v in G.edges:
    A[u, v] = 1
    A[v, u] = 1

linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
mode = "ZERO"
topology = "MIX"
mapping="trivial"
index = "Hellinger2"
opt_score = ["_D"]

properties = {
  "structure" : "Zachary",
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "topology" : topology,
  "mapping" : mapping,
  "index" : index,
  "mode" : mode,
}

NET = TOY(A, linkage, **properties)
NET.set_labels(np.arange(nodes))
H = Hierarchy(
  NET, A, A, np.zeros((34, 34)),
  nodes, linkage, mode,
  lookup=lookup, undirected=T
)
# # Compute quality functions ----
H.BH_features_cpp_no_mu()
## Compute link entropy ----
H.link_entropy_cpp("short", cut=cut)
## Compute la arbre de merde ----
H.la_abre_a_merde_cpp(H.BH[0])
## Compute node entropy ----
H.node_entropy_cpp("short", cut=cut)
# Set labels to network ----
L = colregion(NET, labels=NET.labels)
L.get_regions()
H.set_colregion(L)
HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels)
HS.Z2dict("short")
H.entropy = [
  H.node_entropy, H.node_entropy_H,
  H.link_entropy, H.link_entropy_H
]
# Picasso ----
plot_h = Plot_H(NET, H)
plot_n = Plot_N(NET, H)
# Plot H ----
HS.zdict2newick(HS.tree, weighted=F, on=T)
plot_h.plot_newick_R(
  HS.newick, HS.total_nodes,
  threshold=H.node_entropy[0].shape[0] - np.argmax(H.node_entropy[0]) - 1,
  weighted=F, on=T
)
# plot_h.plot_measurements_Entropy(on=T)
# plot_h.plot_measurements_D(on=T)
# plot_h.plot_measurements_S(on=T)
# plot_h.plot_measurements_SD(on=T)
# plot_h.plot_measurements_X(on=T)
# # # # # Plot N ----
# plot_n.histogram_weight(H.source_sim_matrix, on=T, label="SS")
# plot_n.histogram_weight(H.target_sim_matrix, on=T, label="TS")
# for score in opt_score:
#     print(f"Find node partition using {score}")
#     # Get best K and R ----
#     K, R, HT = get_best_kr_equivalence(score, H)
#     r = R[K == np.min(K)][0]
#     k = K[K == np.min(K)][0]
#     print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
#     rlabels = get_labels_from_Z(H.Z, r)
#     rlabels = skim_partition(rlabels)
# #     # Overlap ----
#     NET.overlap, NET.data_nocs, sizes, rlabels2 = discovery_channel["discovery_7"](H, k, rlabels, index=index, direction="both",  undirected=T)
#     print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
#     cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
#     # Set community structure ----
#     H.set_kr(k, r, score=score)
#     H.set_overlap_labels(NET.overlap, score, "both")
#     H.set_cover(cover, score, "both") 
# #     # Plot H ----
# #     # plot_h.core_dendrogram([r], on=T) #
# #     # plot_h.heatmap_dendro(r, NET.A, on=T)
# #     # plot_h.lcmap_dendro(k, r, undirected=T, on=T) #
#     plot_n.plot_network_kk(
#       H, rlabels2, NET.data_nocs, sizes, H.colregion.labels,
#       ang=80, score=score, undirected=T, on=T, cmap_name="pastel"
#     )

