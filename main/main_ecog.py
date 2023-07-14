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
from modules.colregion import colECoG
from various.data_transformations import maps
from modules.hierarentropy import Hierarchical_Entropy
from networks.ECoG.structure import WAVES
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
nature = "MK1PostGamma"
mode = "ZERO"
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
opt_score = ["_S", "_SD", "_X"]
save_data = F
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = WAVES[nature](
    linkage, mode,
    nlog10 = nlog10,
    lookup = lookup,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut
  )
  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.C, nlog10, lookup, prob
  )
  # Compute Hierarchy ----
  print("Start ELCA")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.C, R, NET.D,
      NET.nodes, linkage, mode, lookup=lookup
    )
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp_no_feat(sp=1)
    ## Compute features nodewise ----
    H.BH_features_cpp_nodewise()
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    ## Update entropy ----
    H.entropy = [H.node_entropy, H.node_entropy_H]
    # Set labels to network ----
    L = colECoG(NET)
    H.set_colregion(L)
    # Save ----
    H.delete_dist_matrix()
    save_class(
      H, NET.pickle_path,
      "hanalysis",
      on=T
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )
  # # Picasso ----
  plot_h = Plot_H(NET, H)
  # HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.regions.AREA_REGION)
  # HS.Z2dict("short")
  # HS.zdict2newick(HS.tree, weighted=F, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=F, on=T)
  # HS.zdict2newick(HS.tree, weighted=T, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=T, on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_SD(on=T)
  # plot_h.plot_measurements_D(on=T)
  # plot_h.plot_measurements_X(on=T)
  plot_n = Plot_N(NET, H)
  logC = NET.C
  logC[logC != 0] = np.log(logC[logC != 0])
  # plot_n.A_vs_dis(logC, s=1, on=T, reg=T)
  # plot_n.histogram_weight(logC, on=T, label="logGC")
  # plot_n.histogram_dist(on=F)
  # plot_n.plot_akis(NET.D, s=1, on=T)
  for score in opt_score:
    print(f"Find node partition using {score}")
    # Get best K and R ----
    K, R = get_best_kr_equivalence(score, H)
    for k, r in zip(K, R):
      # H.set_kr(k, r, score=score)
      print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
      rlabels = get_labels_from_Z(H.Z, r)
      # Overlap ----
      # NET.overlap, NET.data_nocs = H.discovery_3(k, rlabels)
      # H.set_overlap_labels(NET.overlap, score)
      # print(NET.overlap)
      # print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
      # cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
      # H.set_cover(cover, score)
      # Plot N ----
      # plot_n.plot_network_covers(
      #   k, -logC, rlabels,
      #   NET.data_nocs, H.colregion.labels,
      #   score=score, cmap_name="husl", on=T, coords = NET.coords,
      #   modified_labels = H.colregion.regions.AREA_REGION,
      #   not_edges = T
      # )
      # Plot H ----
      # plot_h.core_dendrogram([r], on=T, leaf_font_size=4) #
      plot_h.lcmap_pure([k], labels = rlabels, on=T, font_size=4, linewidth=1)
      plot_h.heatmap_pure(r, logC, on=T, labels = rlabels, font_size=4, linewidth=1) #
      plot_h.heatmap_dendro(r, logC, on=T, font_size=4, linewidth=1)
      plot_h.lcmap_dendro(k, r, on=T, font_size=4, linewidth=1) #
  save_class(
    H, NET.pickle_path,
    "hanalysis"
  )
  print("End!")
  # #@@ Todo: