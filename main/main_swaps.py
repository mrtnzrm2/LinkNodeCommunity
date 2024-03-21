# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Allias booleans ----
T = True
F = False
# Personal libs ----
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from various.data_transformations import maps
from networks.structure import MAC
from networks.swapnet import SWAPNET
from various.network_tools import *
# Declare global variables ----
__iter__ = 0
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
run = T
inkage = "single"
mode = "ZERO"
alpha = 0.
structure = "FLN"
distance = "tracto16"
nature = "original"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "s"
imputation_method = ""
opt_score = ["_S"]

save_data = T
# Declare global variables DISTBASE ----
__inj__ = 57
__nodes__ = 57
_total_nodes = 106
__version__ = "57d106"
__model__ = "1k"

if __name__ == "__main__":
  # Create EDR network ----

  REF = MAC[f"MAC{__inj__}"](
    linkage, mode,
    structure = structure,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    distance = distance,
    nature = nature,
    model = imputation_method,
    inj= __inj__,
    topology= topology,
    mapping=mapping,
    index=index,
    cut = cut,
    b=bias, alpha=alpha,
    discovery = discovery
  )

  NET = SWAPNET(
   __inj__,
    _total_nodes,
    linkage,
    mode, __iter__,
    structure = structure,
    version = __version__,
    topology=topology,
    nature = nature,
    distance = distance,
    model = __model__,
    mapping=mapping,
    index=index,
    nlog10 = nlog10, lookup = lookup,
    cut=cut, b=bias, discovery=discovery
  )
  NET.C, NET.A = REF.C, REF.A
  NET.D = REF.D
  NET.struct_labels = REF.struct_labels
  # NET.create_csv_path()
  # NET.create_pickle_path()
  NET.create_plot_path()
  # Create network ----
  print("Create random graph")
  NET.random_one_k(run=run, on_save_csv=F)
  # Transform data for analysis ----
  R, lookup, shift = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  # Add color ----
  L = colregion(REF, labels_name=f"labels{__inj__}")
  NET.set_colregion(L)
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A[:, :NET.nodes], R[:, :NET.nodes],
      NET.D, __nodes__, linkage, mode, lookup=lookup, alpha=alpha
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    # Set entropy ----
    H.entropy = [
      H.node_entropy, H.node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    H.set_colregion(L)
    H.delete_dist_matrix()
    save_class(
      H, NET.pickle_path,
      "hanalysis",
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )
  # Plot H ----
  plot_h = Plot_H(NET, H)
  HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  HS.Z2dict("short")
  HS.zdict2newick(HS.tree, weighted=F, on=T)
  plot_h.plot_newick_R(HS.newick, weighted=F, on=T)
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  # Plot N ----
  # plot_n = Plot_N(NET, H)
  # plot_n.A_vs_dis(NET.A, s=5, on=F)
  # plot_n.A_vs_dis(NET.C, s=5, name="count", on=F)
  # plot_n.histogram_weight(on=F)
  # plot_n.projection_probability(
  #   NET.A[:, :__nodes__], bins=12, on=F
  # )
  # plot_n.plot_akis(
  #   NET.D, s=5, on=F
  # )
  # for score in opt_score:
  #   print(f"Find node partition using {score}")
  #   K, R = get_best_kr_equivalence(score, H)
  #   r = R[K == np.min(K)]
  #   k = K[K == np.min(K)]
  #   print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
  #   ## Take a look in case of SLN ----
  #   rlabels = get_labels_from_Z(H.Z, r)
  #   NET.overlap, _ = H.discovery_3(k, rlabels)
  #   H.set_overlap_labels(NET.overlap, score)
  #   ## Single linkage ----
  #   plot_h.core_dendrogram([r], on=F)
  #   plot_h.heatmap_pure(
  #     r, on=F, labels = rlabels,
  #     score="_"+score
  #   )
  #   plot_h.heatmap_dendro(
  #     r, on=F, score="_"+score
  #   )
  #   plot_h.lcmap_dendro(
  #     [k], on=F, score="_"+score
  #   )
  print("End!")
  # #@@ Todo: