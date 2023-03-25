# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from various.data_transformations import maps
from networks.structure import MAC
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
mode = "ALPHA"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "R2"
index  = "jacw"
bias = float(1e-5)
opt_score = ["_maxmu", "_X"]
save_data = T
version = 220830
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC(
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
  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup
    )
    ## Compute features ----
    H.BH_features_parallel()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET)
    H.set_colregion(L)
    # Save ----
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
  # Entropy ----
  HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  HS.Z2dict("short")
  node_entropy = HS.S(HS.tree)
  node_entropy_H = HS.S_height(HS.tree)
  H.entropy = [
    node_entropy, node_entropy_H,
    H.link_entropy, H.link_entropy_H
  ]
  # # Picasso ----
  plot_h = Plot_H(NET, H)
  HS.zdict2newick(HS.tree, weighted=F, on=T)
  plot_h.plot_newick_R(HS.newick, weighted=F, on=F)
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  plot_h.plot_newick_R(HS.newick, weighted=T, on=F)
  plot_h.plot_measurements_Entropy(on=F)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_mu(on=T)
  plot_h.plot_measurements_X(on=T)
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(R, s=5, on=F, reg=T)
  plot_n.projection_probability(
    NET.C, bins=12, on=F
  )
  plot_n.histogram_weight(R, on=F)
  plot_n.histogram_dist(on=F)
  plot_n.plot_akis(NET.D, s=5, on=F)
  for score in opt_score:
    print(f"Find node partition using {score}")
    # Get best K and R ----
    K, R = get_best_kr(score, H)
    r = R[K == np.min(K)][0]
    k = K[K == np.min(K)][0]
    H.set_kr(k, r, score=score)
    print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
    rlabels = get_labels_from_Z(H.Z, r)
    # Overlap ----
    NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
    cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
    H.set_cover(cover, score)
    # Plot H ----
    plot_h.core_dendrogram([r], on=F) #
    plot_h.lcmap_pure([k], labels = rlabels, on=F)
    plot_h.heatmap_pure(r, on=F, labels = rlabels) #
    plot_h.heatmap_dendro(r, on=F)
    plot_h.lcmap_dendro([k], on=F) #
    plot_h.flatmap_dendro(
      NET, [k], [r], on=F, EC=T #
    )
  save_class(
    H, NET.pickle_path,
    "hanalysis"
  )
  print("End!")
  # #@@ Todo: