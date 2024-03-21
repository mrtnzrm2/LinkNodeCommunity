# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libraries ----
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import itertools
# Import network libraries ----
from plotting_modules.plotting_serial import PLOT_S
from plotting_modules.plotting_o_serial import PLOT_OS
from modules.hierarentropy import Hierarchical_Entropy
from modules.colregion import colregion
from networks.structure import MAC
from various.network_tools import read_class, skim_partition
from various.clustering_tools import discover_overlap_nodes
# Declare iter variables ----
topologies = ["MIX"]
bias = [0]
mode = ["ZERO"]
list_of_lists = itertools.product(
  *[topologies, bias, mode]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables NET ----
MAXI = 100
linkage_name = "single"
nlog10 = T
lookup = F
prob = F
cut = F
run = T
structure = "FLN"
distance = "tracto16"
nature = "original"
mapping = "trivial"
index  = "Hellinger2"
imputation_method = ""
opt_score = ["_S"]
alpha = 0.
# Statistic test ----
alternative = "less"
# Declare global variables DISTBASE ----
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__version__ = "57d106"
if __name__ == "__main__":
  for topology, bias, mode in list_of_lists:
    bias = float(bias)
    # Print summary ----
    print("For NET parameters:")
    print(
      "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup : {}".format(
        linkage_name, mode, opt_score, nlog10, lookup
      )
    )
    print("For imputation parameters:")
    print(
      "nature: {}\nmodel: {}".format(
        nature, imputation_method
      )
    )
    print("Random network and statistical paramteres:")
    print(
      "nodes: {}\ninj: {}\nalternative: {}".format(
        str(__nodes__),str(__inj__), alternative
      )
    )
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    data = read_class(
        "../pickle/RAN/shuffle/MAC/{}/{}/{}/{}/{}/{}/{}/{}".format(
          __version__,
          structure,
          distance,
          f"{linkage_name.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
          mode,
          f"{topology}_{index}_{mapping}",
          f"b_{bias}",
          "discovery_7"
          # f"alpha_{alpha:.2f}"
        ),
        "series_{}".format(MAXI)
      )
    if isinstance(data, int): continue
    NET = MAC[f"MAC{__inj__}"](
      linkage_name, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = __version__,
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
      discovery = "discovery_7"
    )
    L = colregion(NET, labels_name=f"labels{__inj__}")
    A = NET.A[:NET.nodes, :NET.nodes].copy()
    A[A > 0] = -np.log(A[A > 0])
    # Hierarchical association ----
    castle = np.mean(data.hierarchical_association, axis=0)
    Zmean = linkage(squareform(castle))
    # Newick ----
    s = Hierarchical_Entropy(Zmean, NET.nodes, NET.struct_labels[:NET.nodes])
    s.Z2dict("short")
    treeh = s.zdict2newick(s.tree, weighted=T)
    # Plotting ----
    plot_o = PLOT_OS(data)
    plot_o.plot_heatmap_Z(castle, on=T)
    plot_o.plot_newick(treeh, L, width=5, height=10, fontsize=9, on=T)
    for score in opt_score:
      Rmean = data.kr["R"].loc[data.kr["score"] == score].mean()
      Rmean = int(Rmean)
      plot_o.core_dendrogram(Zmean, [Rmean], leaf_font_size=10, on=T)
      rlabels = cut_tree(Zmean, n_clusters=Rmean).ravel()
      rlabels = skim_partition(rlabels)
      for direction in ["source", "target", "both"]:
        _, data_nocs, noc_sizes, rlabels2  = discover_overlap_nodes(
          NET.A, np.sqrt(1-data.source_sim_matrix), np.sqrt(1-data.target_sim_matrix),
          rlabels, NET.struct_labels[:NET.nodes], direction=direction
        )
        plot_o.plot_network_covers(
          A, rlabels2,
          data_nocs, noc_sizes,
          score=score, direction=direction, cmap_name="hls", figsize=(8,8), on=T
        )
