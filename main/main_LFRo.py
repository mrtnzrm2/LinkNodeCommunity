# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Allias booleans ----
T = True
F = False
#Import libraries ----
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from networks.overlapping import OVERLAPPING
from modules.colregion import colregion
from numpy import zeros
from various.network_tools import *
# Declare global variables ----
__iter__ = 1
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
__mode__ = "ZERO"
alpha = 0.
opt_score = ["_X" ,"_D", "_S", "_SD"]
save_datas = T
# Overlapping WDN paramters ----
opar = {
  "-N" : "{}".format(
    str(__nodes__)
  ),
  "-k" : "7",
  "-maxk" : "20",
  "-mut" : "0.1",
  "-muw" : "0.01",
  "-beta" : "3",
  "-t1" : "2",
  "-t2" : "1",
  "-nmin" : "6",
  "-nmax" : "25",
  "-on" : "10",
  "-om" : "2"
}
if __name__ == "__main__":
  # Create EDR network ----
  NET = OVERLAPPING(
    __iter__,
    linkage,
    __mode__,
    parameters=opar,
    nlog10=nlog10, lookup=lookup,
    topology=topology,
    mapping=mapping,
    index=index,
    cut=cut
  )
  NET.create_plot_path()
  NET.create_pickle_path()
  # Create network ----
  print("Create random graph")
  NET.random_BN_overlap_cpp(
    run=run, on_save_pickle=T
  )
  if np.sum(np.isnan(NET.A)) > 0:
    raise RuntimeError(
      "LFB failed to create the network with the desired properties."
    )
  NET.col_normalized_adj(on=F)
  number_of_communities = len(np.unique(NET.labels))
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Create colregions ----
  L = colregion(NET)
  NET.set_colregion(L)
  # Save ----
  if save_datas:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, NET.A, zeros(NET.A.shape),
      __nodes__, linkage, __mode__, alpha=alpha
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    H.set_colregion(L)
    H.delete_dist_matrix()
    save_class(
      H, NET.pickle_path, "hanalysis_{}".format(H.subfolder),
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path, "hanalysis_{}".format(NET.subfolder)
    )
  # Plot H ----
  plot_h = Plot_H(NET, H)
  # plot_h.plot_measurements_D(on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_mu(on=T)
  # plot_h.plot_measurements_X(on=T)
  # plot_h.heatmap_pure(
  #    0, np.log(1 + NET.A), score = "_GT_{}".format(number_of_communities), linewidth=0.5, 
  #    font_size=1, labels = NET.labels, on=T
  # )
  for score in opt_score:
    # Find best k partition ----
    K, R = get_best_kr_equivalence(score, H)
    for ii, kr in enumerate(zip(K, R)):
      k, r = kr
      rlabels = get_labels_from_Z(H.Z, r)
      # Check labels safety ----
      if np.nan in rlabels:
          print("Warning: Impossible node dendrogram")
          break
      nocs, noc_covers, _ = H.discovery_channel["discovery_7"](H, k, rlabels)
      #Prints ----
      nmi = AD_NMI_overlap(
        NET.labels, rlabels, NET.overlap, noc_covers, on=T
      )
      sen, sep = NET.overlap_score_discovery(
        k, nocs, H.colregion.labels[:H.nodes], on=T
      )
      omega = NET.omega_index(
        rlabels, noc_covers, H.colregion.labels[:H.nodes], on=T
      )
      ## Plots ---
      # plot_h.core_dendrogram(
      #   [r], on=T, score="_"+score, remove_labels=True
      # )
      # plot_h.heatmap_pure(
      #    r, np.log(1+NET.A), on=T, labels = rlabels, name=f"{r}_{nmi:.4f}",
      #    linewidth=1, font_size=1
      # )
      # plot_h.lcmap_dendro(
      #    k, np.log(1+NET.A), score="_"+score, on=T,
      #    linewidth=0.5, remove_labels=True
      # )
  print("End!")