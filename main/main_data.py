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
# from modules.sign.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import *
from various.data_transformations import maps
from networks.structure import STR
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
nature = "original"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
bins = 12
discovery = "discovery_7"
opt_score = ["_S"]
sln = T

__nodes__ = 40
__inj__ = "40"
version = f"40"+"d"+"91"
distance = "MAP3D"
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
      __nodes__, linkage, mode, lookup=lookup,
      index=index,
      is_sln=False,
      # SLN=NET.SLN
      #  architecture="all"
    )

    # Compute similarity matrix ----
    H.similarity_linksim_edgelist()
    H.dist_edgelist = H.linksim_edgelist.copy()
    H.dist_edgelist[:, 2] = 1 - H.dist_edgelist[:, 2]
    H.H = H.get_hierarchy_edgelist(max_dist=1)

    print(">>> Hierarchy created")
    ## Compute features ----
    H.process_features_edgelist()
    ## Compute la arbre de merde ----
    H.node_community_hierarchy_edgelist(H.BH[0])
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__inj__}")
    H.set_colregion(L)
    # Save ----
    H.delete_linksim_matrix()
    H.delete_dist_matrix()
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )

  # # Picasso ----
  plot_h = Plot_H(NET, H)
  plot_n = Plot_N(NET, H)
  HS = Hierarchical_Entropy(H.Z, H.nodes, labels=NET.struct_labels[:NET.nodes])
  HS.Z2dict("fit")
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  plot_h.plot_newick_R(
    HS.newick, HS.N,
    root_position=1 - H.Z[:, 2][-1],
    # threshold=1 - H.Z[:, 2][-1],
    # threshold=H.node_entropy[0].shape[0] - np.argmax(H.node_entropy[0]) - 1,
    weighted=T, on=T
  )
  # plot_h.plot_newick_Rpretty(HS.newick, HS.N, NET.csv_path, root_position=1- H.Z[:, 2][-1], weighted=T, on=F)

  # plot_h.plot_measurements_Entropy(on=T)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_SD(on=T)
  # plot_h.plot_measurements_CC(on=T)

  RN = NET.A[:__nodes__, :].copy()
  RN[RN > 0] = -np.log(RN[RN > 0])
  np.fill_diagonal(RN, 0.)

  RW = NET.A.copy()
  RW[RW > 0] = -np.log(RW[RW > 0])
  np.fill_diagonal(RW, 0.)

  RW10 = NET.A.copy()
  RW10[RW10 > 0] = -np.log10(RW10[RW10 > 0])
  # np.fill_diagonal(RW10, 0.)

  # plot_n.A_vs_dis(-RW, s=10, on=F, reg=T)
  # plot_n.projection_probability(NET.CC, "EXPTRUNC" , bins=bins, on=F)
  # plot_n.histogram_dist(on=True)
  # plot_n.histogram_weight(-RW10, label=r"$\log10(p(i,j))$", suffix="log10_p", on=F)
  plot_n.plot_akis(NET.D, s=5, on=F)

  for SCORE in opt_score:
    # Get best K and R ----
    K, R, TH = get_best_kr_equivalence(SCORE, H)

    # K = [get_k_from_equivalence(13, H)]
    # R = [13]

    for k, r, th in zip(K, R, TH):
      print(f"Find node partition using {SCORE}")
      print("Best K: {}\t R: {}: and TH : {}\t Score: {}".format(k, r, th, SCORE))
      H.set_kr(k, r, score=SCORE)
      rlabels = get_labels_from_Z(H.Z, r)
      rlabels = skim_partition(rlabels)

      if -1 in rlabels:
        print(">>> Single community nodes:")
        SCN = NET.struct_labels[:NET.nodes][np.where(rlabels == -1)[0]]
        print(SCN, "\n")

      # Plot H ----
      plot_h.core_dendrogram([r], leaf_font_size=12, on=F) #
      plot_h.heatmap_dendro(
        r, -RW10, linewidth=2.5, score="FLNe",
        cbar_label=r"$\log_{10}$" + "FLNe",
        fontsize = 25, suffix="5", on=F
      )
      plot_h.lcmap_dendro(k, r, on=F, font_size = 25, linewidth=2.5) #
      # plot_h.lcmap_pure()
      # plot_h.threshold_color_map(r, th, index=index, score=SCORE, font_size = 12, on=F)
      
      # Overlap ----
      for direction in ["both"]: # ,  "source", "target",
        print("***", direction)
        NET.overlap, NET.data_nocs, noc_sizes,  rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)


        print(">>> Areas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
        print(">s>> Single areas assigned to one cover:\n", {k : rlabels2[np.where(NET.struct_labels[:NET.nodes] == k)][0] for k in SCN if rlabels2[np.where(NET.struct_labels[:NET.nodes] == k)][0] != -1})
        cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])

        
        # print(cover)

        # #  FRP  ACAd  ACAv  PL  ILA  ORBl  ORBm  ORBvl  AId  AIv  AIp  GU  VISC  TEa  PERI  ECT  SSs  SSp-bfd  SSp-tr  SSp-ll  SSp-ul  SSp-un  SSp-n  SSp-m  MOp  MOs  VISal  VISl  VISp  VISpl  VISli  VISpor  VISrl  VISa  VISam  VISpm  RSPagl  RSPd  RSPv  AUDd  AUDp  AUDpo  AUD
        # AllenLabels = [0] * 8 + [1] * 8 + [2] * 10 +[3] * 7 + [4] * 6 + [5] * 4
        # Allen_nocs = {}

        # Allen_cover = omega_index_format(AllenLabels, Allen_nocs, NET.struct_labels[:NET.nodes])

        # print(">>>>", omega_index(cover, Allen_cover))


        # if direction == "both" and sln:

        #   plot_h.heatmap_sln_dendro(
        #     r, -RN[:NET.nodes, :NET.nodes], NET.SLN[:NET.nodes, :NET.nodes],
        #     on=F, score="SLN", cbar_label="SLN", center=0.5, font_size = 12, suffix=""
        #   )
          
        #   data_sln = H.get_data_firstmerge(NET.SLN, cover, NET.struct_labels)
        #   SLN_cover_matrix = H.get_sln_matrix(data_sln, cover)

        #   cover, color_order = H.align_sln_covers_R(SLN_cover_matrix, cover, NET.csv_path)
        #   cover = H.reorder_nodes_in_cover_H2(cover, NET.struct_labels[:NET.nodes])

        #   data_sln = H.get_data_firstmerge(NET.SLN, cover, NET.struct_labels, betas=NET.beta)
        #   SLN_cover_matrix = H.get_sln_matrix(data_sln, cover)
          
        #   plot_n.plot_cover_items(cover, on=F)
        #   H.set_data_sln_matrix(SLN_cover_matrix)
        #   H.set_data_sln(data_sln)
        #   plot_h.hitsplot_sln_corr_covers(data_sln, cover, on=False)
        #   plot_h.heatmap_sln_corr_covers(data_sln, cover, on=True)
        #   plot_h.sln_trace(data_sln, cover, xlabel="SLN_BB", ylabel="SLN", suffix="BB", on=False)
        #   plot_h.sln_offdiagonal(data_sln, cover, on=False)
          # plot_h.sln_matrix(data_sln, cover, cbarlabel="Empirical SLN", on=False)


        # else:
          # H.set_data_sln(0)
          # H.set_data_sln_matrix(0)
 
        # rlabels2[rlabels2 != -2] = 0
        # cover_art = {}

        plot_n.plot_network_covers(
          k, r, RN, rlabels2, rlabels,
          NET.data_nocs,
          # cover_art,
          noc_sizes, H.colregion.labels[:H.nodes], ang=0,
          # color_order=color_order,
          score=SCORE, direction=direction, spring=F, font_size=20,
          scale=0.45,
          suffix="small", cmap_name="deep", not_labels=F, on=F#, figsize=(8,8)
        )

      # cover_art = {}
      # rlabels = [0 for i in np.arange(len(rlabels))]
      plot_n.plot_network_covers(
          k, r,  RN, rlabels2, rlabels,
          NET.data_nocs,
          # cover_art, 
          noc_sizes, H.colregion.labels[:H.nodes], ang=0,
          # color_order=color_order,
          scale=0.5, font_size=16.5,
          # exchange_K=[(0, 1), (2, 3), (2, 4), (2, 1)],
          # exchange_K=[(0, 2), (2, 4), (2, 6)],
          score=SCORE, direction=direction,
          suffix="", cmap_name="deep", on=T#, figsize=(8,8)
        ) 

  #       # plot_n.distance_cover_boxplot(rlabels2, c over, direction=direction, index=index, on=F)
  #       # # Flat map ---
  #       # plot_h.flatmap_dendro(
  #       #   NET, k, r, rlabels2, direction=direction, on=T, EC=T, cmap_name="hls" #
  #       # )

  #     H.set_rlabels(rlabels2, SCORE, direction)
  #     H.set_overlap_labels(NET.overlap, SCORE, direction)
  #     H.set_cover(cover, SCORE, direction)

  # #   # index_distance = H.target_sim_matrix.copy()
  # #   # np.fill_diagonal(index_distance, np.nan)
  # #   # index_distance = -2 * np.log(index_distance)
  # #   # max_index_distance = np.nanmax(index_distance[index_distance < np.inf])
  # #   # for i, area in enumerate(NET.struct_labels[:NET.nodes]):
  # #   #   values = index_distance[i, :]
  # #   #   plot_h.flatmap_index(NET, area, values, max_value=max_index_distance, index_name=index, on=T)
  # plot_h.flatmap_regions(
  #     NET, k, r, rlabels2, direction=direction, on=F, EC=T, cmap_name="deep" #
  #   ) 
  
  save_class(
    H, NET.pickle_path,
    "hanalysis", on=T
  )
  print("End!")
  # #@@ Todo: