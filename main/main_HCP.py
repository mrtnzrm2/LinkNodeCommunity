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
from plotting_modules.plotting_HCP import Plot_HCP
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps
from networks.HCP.HCP import HCP
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject_id = None
structure = "Cor"
mode = "ZERO"
nature = "original"
topology = "MIX"
mapping = "signed_trivial"
index  = "Hellinger2"
discovery = "discovery_7"
architecture = "product-positive"
opt_score = ["_S"]
undirected_network = F
undirected_merde = 1
nodetimeseries = "50"
save_data = F
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = HCP(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    nature = nature,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    nodetimeseries = nodetimeseries,
    discovery = discovery,
    undirected = undirected_merde,
    architecture = architecture,
    subject_id=subject_id
  )
  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=0
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, np.zeros(NET.A.shape),
      NET.nodes, linkage, mode, lookup=lookup,
      undirected=undirected_network,
      architecture=architecture
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0], undirected=undirected_merde)
    # Set labels to network ----
    L = colregion(NET)
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
  # Picasso ----
  plot_h = Plot_HCP(NET, H)
  plot_n = Plot_N(NET, H)
  # print(H.Z)
  HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  HS.Z2dict("short")
  tree = HS.zdict2newick(HS.tree, weighted=F)
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  # plot_h.plot_newick_R(HS.newick, HS.total_nodes, weighted=T, on=T)
  # plot_h.plot_newick_R_PIC(tree, NET.picture_path, on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_n.histogram_weight(NET.A, label="Cor")

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
      # plot_h.core_dendrogram([r], leaf_font_size=10, on=T) #
      plot_h.heatmaply_dendro(r, NET.A, on=T, score="Cor", font_size = 10, font_color="white", cmap="RdBu_r", center=0)
      plot_h.lcmaply_dendro(k, r, on=T, font_size = 10, undirected=undirected_network, font_color="white") #
      # plot_h.threshold_color_map(r, th, index=index, score=SCORE, font_size = 10, on=T)
      # plot_h.plotly_nodetimeseries_Z(r, rlabels, small_set=T, on=F)
      
      # Overlap ----
      for direction in ["both"]:
        print("***", direction)
        NET.overlap, NET.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](
          H, k, rlabels, direction=direction, index=index, undirected=undirected_network
        )
        print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
        cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])
        H.set_cover(cover, SCORE, direction)

  #       # Netowk ----
  #       R = NET.A[:NET.nodes, :]
  #       R[R < 0] = -R[R < 0]
  #       R = 1 - R/(np.nanmax(R)+1e-2)
  #       np.fill_diagonal(R, 0)
  #       plot_n.plot_network_covers(
  #         k, R, rlabels2,
  #         NET.data_nocs, noc_sizes, H.colregion.labels[:H.nodes],
  #         score=SCORE, direction=direction, cmap_name="hls", on=T, figsize=(8,8),
  #         undirected=undirected_network, scale=5, font_size=1
  #       )
        H.set_rlabels(rlabels2, SCORE, direction)
        H.set_overlap_labels(NET.overlap, SCORE, direction)
        H.set_cover(cover, SCORE, direction)
  # save_class(
  #   H, NET.pickle_path,
  #   "hanalysis", on=T
  # )
  print("End!")