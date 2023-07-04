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
cut = T
nature = "MK1PreGamma"
mode = "BETA"
topology = "MIX"
mapping = "trivial"
index  = "D1_2_2"
opt_score = ["_X", "_S"]
save_data = T
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
    L = colECoG(NET)
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
  # # Picasso ----
  plot_h = Plot_H(NET, H)
  HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.regions.AREA_REGION)
  HS.Z2dict("short")
  HS.zdict2newick(HS.tree, weighted=F, on=T)
  plot_h.plot_newick_R(HS.newick, weighted=F, on=T)
  HS.zdict2newick(HS.tree, weighted=T, on=T)
  plot_h.plot_newick_R(HS.newick, weighted=T, on=T)
  plot_h.plot_measurements_Entropy(on=T)

  plot_h.plot_measurements_D(on=F)
  plot_h.plot_measurements_mu(on=F)
  plot_h.plot_measurements_X(on=F)
  plot_n = Plot_N(NET, H)
  logC = NET.C
  logC[logC != 0] = np.log(logC[logC != 0])
  plot_n.A_vs_dis(logC, s=1, on=T, reg=T)
  plot_n.histogram_weight(logC, on=T, label="logGC")
  plot_n.histogram_dist(on=F)
  plot_n.plot_akis(NET.D, s=1, on=T)
  for score in opt_score:
    print(f"Find node partition using {score}")
    # Get best K and R ----
    K, R = get_best_kr_equivalence(score, H)
    r = R[K == np.min(K)][0]
    k = K[K == np.min(K)][0]
    H.set_kr(k, r, score=score)
    print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
    rlabels = get_labels_from_Z(H.Z, r)
    # Overlap ----
    NET.overlap, NET.data_nocs = H.discovery_2(k, rlabels, rho=1.2, sig=0.8, fun=np.log)
    H.set_overlap_labels(NET.overlap, score)
    print(NET.overlap)
    print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
    cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
    H.set_cover(cover, score)
    # Plot H ----
    plot_h.core_dendrogram([r], on=T) #
    plot_h.lcmap_pure([k], labels = rlabels, on=F)
    plot_h.heatmap_pure(r, C, on=T, labels = rlabels) #
    plot_h.heatmap_dendro(r, C, on=T)
    plot_h.lcmap_dendro(k, r, on=T) #
    plot_h.flatmap_dendro(
      NET, [k], [r], on=T, EC=T #
    )
  save_class(
    H, NET.pickle_path,
    "hanalysis"
  )
  print("End!")
  # #@@ Todo: