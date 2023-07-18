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
prob = F
cut = F
structure = "LN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
bias = 0.
alpha = 0.
discovery = "discovery_4"
opt_score = ["_S", "_X", "_SD"]
save_data = T
version = "57d106"
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
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
    discovery = discovery
  )
  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.C, nlog10, lookup, prob, b=bias
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.C, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup
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
    L = colregion(NET, labels_name=f"labels{__inj__}")
    H.set_colregion(L)
    # Save ----
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
  # # Picasso ----
  plot_h = Plot_H(NET, H)
  plot_n = Plot_N(NET, H)
  # HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  # HS.Z2dict("short")
  # HS.zdict2newick(HS.tree, weighted=F, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=F, on=T)
  # HS.zdict2newick(HS.tree, weighted=T, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=T, on=T)
  # plot_h.plot_measurements_Entropy(on=T)
  # plot_h.plot_measurements_D(on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_SD(on=T)
  # plot_h.plot_measurements_X(on=T)
  # plot_n.A_vs_dis(np.log(1 + NET.C), s=5, on=F, reg=T)
  # plot_n.projection_probability(
  #   NET.C, "EXPMLE" , bins=12, on=T
  # )
  # plot_n.histogram_dist(on=F)
  # plot_n.plot_akis(NET.D, s=5, on=T)
  for SCORE in opt_score:
    # Get best K and R ----
    K, R = get_best_kr_equivalence(SCORE, H)
    for k, r in zip(K, R):
      print(f"Find node partition using {SCORE}")
      print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, SCORE))
      H.set_kr(k, r, score=SCORE)
      rlabels = get_labels_from_Z(H.Z, r)

      # Overlap ----
      NET.overlap, NET.data_nocs, noc_covers  = H.discovery_channel[discovery](H, k, rlabels)
      H.set_overlap_labels(NET.overlap, SCORE)
      print(NET.overlap)
      print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
      cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
      H.set_cover(cover, SCORE)
  #     # Plot N ----
      plot_n.plot_network_covers(
        k, np.log(1 + NET.C[:__nodes__, :]), rlabels,
        NET.data_nocs, noc_covers, H.colregion.labels[:H.nodes],
        score=SCORE, cmap_name="husl", on=T
      )
  #     # Plot H ----
  #     plot_h.core_dendrogram([r], on=T) #
  #     plot_h.lcmap_pure([k], labels = rlabels, on=T)
  #     plot_h.heatmap_pure(r, np.log10(1+NET.C), on=T, labels = rlabels, score='LN') #
  #     plot_h.heatmap_dendro(r, np.log(NET.A), on=T, score="FLN", font_size=20)
  #     plot_h.lcmap_dendro(k, r, on=T, font_size = 20) #
  #     plot_h.flatmap_dendro(
  #       NET, [k], [r], on=T, EC=T #
  #     )
  # NET.overlap = np.char.array(["v2pcuf", "opro", "f1"])
  # plot_h.flatmap_dendro(
  #       NET, [0], [0], on=T, EC=T #
  # )
  # save_class(
  #   H, NET.pickle_path,
  #   "hanalysis", on=T
  # )
  print("End!")
  # #@@ Todo: