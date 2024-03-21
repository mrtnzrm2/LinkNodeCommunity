# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.sign.hierarmerge import Hierarchy
# from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import *
from networks.structure import STR
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject_id = None
subject = "MAC"
structure = "MULTIPLEX_SI"
mode = "ZERO"
distance = "tracto16"
nature = "original"
# imputation_method = "RF2"
topology = "MIX"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "discovery_7"
opt_score = ["_S"]
save_data = T
__nodes__ = 49
__inj__ = f"{__nodes__}"
version = "220617"
mapping = "multiplexed_colnormalized" # multiplexed_colnormalized column_normalized_mapping
architecture = "all"
opt_score = ["_S"]
save_data = T
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = STR[f"{subject}{__inj__}"](
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
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
  # R, lookup, _ = column_normalized_mapping(NET.SN)
  # R, lookup, _ = column_normalized_mapping(NET.IN)
  R, lookup, _ = multiplexed_colnormalized_mapping(0, NET.SN, NET.IN)
  # R, lookup, _ = trivial_mapping(NET.A)
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, NET.D,
      NET.nodes, linkage, mode, lookup=lookup,
      architecture=architecture
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__inj__}")
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
  plot_n = Plot_N(NET, H)
  HS = Hierarchical_Entropy(H.Z, H.nodes, NET.struct_labels[:NET.nodes])
  HS.Z2dict("short")
  # HS.zdict2newick(HS.tree, weighted=F, on=T)
  # plot_h.plot_newick_R(
  #   HS.newick, HS.total_nodes,
  #   threshold=H.node_entropy[0].shape[0] - np.argmax(H.node_entropy[0]) - 1,
  #   weighted=F, on=T
  # )
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  plot_h.plot_newick_R(HS.newick, HS.total_nodes, weighted=T, on=T)
  # plot_h.plot_measurements_Entropy(on=T)
  # plot_h.plot_measurements_D(on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_SD(on=T)
  # plot_h.plot_measurements_X(on=T)

  # plot_n.projection_probability(NET.C, "EXPMLE" , bins=12, on=T)
  # plot_n.histogram_dist(on=True)
  RN = NET.A[:__nodes__, :].copy()
  RN[RN > 0] = -np.log(RN[RN > 0])
  np.fill_diagonal(RN, 0.)

  RW = NET.A.copy()
  RW[RW > 0] = -np.log(RW[RW > 0])
  np.fill_diagonal(RW, 0.)

  plot_n.A_vs_dis(-RW, s=10, on=T, reg=T)
  plot_n.histogram_weight(-RN, label=structure, on=T)
  plot_n.plot_akis(NET.D, s=5, on=T)

  for SCORE in opt_score:
    # Get best K and R ----
    K, R, TH = get_best_kr_equivalence(SCORE, H)
    for k, r, th in zip(K, R, TH):
      print(f"Find node partition using {SCORE}")
      print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, SCORE))
      H.set_kr(k, r, score=SCORE)
      rlabels = get_labels_from_Z(H.Z, r)
      rlabels = skim_partition(rlabels)
       # Plot H ----
      plot_h.core_dendrogram([r], leaf_font_size=8, on=T) #
      plot_h.heatmap_dendro(r, -RN, on=T, score=structure, font_size = 12)
      plot_h.lcmap_dendro(k, r, on=T, font_size = 12) #
      plot_h.threshold_color_map(r, th, index=index, score=SCORE, font_size = 12, on=T)
      
      # Overlap ----
      for direction in ["source", "target", "both"]:
        print("***", direction)
        NET.overlap, NET.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)
        print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
        cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])
        H.set_cover(cover, SCORE, direction)


        plot_n.plot_network_covers(
          k, RN, rlabels2,
          NET.data_nocs, noc_sizes, H.colregion.labels[:H.nodes], ang=0,
          score=SCORE, direction=direction, cmap_name="hls", on=T#, figsize=(8,8)
        )

        # plot_n.distance_cover_boxplot(rlabels2, cover, direction=direction, index=index, on=T)
        # Flat map ---
        # plot_h.flatmap_dendro(
        #   NET, k, r, rlabels2, direction=direction, on=T, EC=T, cmap_name="hls" #
        # )

        H.set_rlabels(rlabels2, SCORE, direction)
        H.set_overlap_labels(NET.overlap, SCORE, direction)
        H.set_cover(cover, SCORE, direction)

        print("\n****\n")

    # index_distance = H.target_sim_matrix.copy()
    # np.fill_diagonal(index_distance, np.nan)
    # index_distance = -2 * np.log(index_distance)
    # max_index_distance = np.nanmax(index_distance[index_distance < np.inf])
    # for i, area in enumerate(NET.struct_labels[:NET.nodes]):
    #   values = index_distance[i, :]
    #   plot_h.flatmap_index(NET, area, values, max_value=max_index_distance, index_name=index, on=T)
  # plot_h.flatmap_regions(
  #     NET, k, r, rlabels2, direction=direction, on=T, EC=F, cmap_name="hls" #
  #   ) 
  save_class(
    H, NET.pickle_path,
    "hanalysis", on=T
  )
  print("End!")