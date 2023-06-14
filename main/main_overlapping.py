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
__iter__ = 3
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
topology = "TARGET"
mapping = "trivial"
index  = "bsim"
__mode__ = "ALPHA"
alpha = 0.
opt_score = ["_maxmu", "_X", "_D"]
save_datas = T
# Overlapping WDN paramters ----
opar = {
  "-N" : "{}".format(
    str(__nodes__)
  ),
  "-k" : "7.0",
  "-maxk" : "30.0",
  "-mut" : "0.1",
  "-muw" : "0.1",
  "-beta" : "2.5",
  "-t1" : "2",
  "-t2" : "1",
  "-nmin" : "2",
  "-nmax" : "10",
  "-on" : "10",
  "-om" : "3"
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
  NET.set_alpha([6, 20])
  NET.create_plot_path()
  NET.create_pickle_path()
  # Create network ----
  print("Create random graph")
  NET.random_WDN_overlap_cpp(
    run=run, on_save_pickle=T
  )
  if np.sum(np.isnan(NET.A)) > 0:
    raise RuntimeError(
      "LFB failed to create the network with the desired properties."
    )
  NET.col_normalized_adj(on=F)
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
    H.BH_features_parallel()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    H.set_colregion(L)
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
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_mu(on=T)
  plot_h.plot_measurements_X(on=T)
  plot_h.heatmap_pure(
    0, score = "_GT", labels = NET.labels, on=T
  )
  for score in opt_score:
    # Find best k partition ----
    K, R = get_best_kr(score, H)
    for ii, kr in enumerate(zip(K, R)):
      k, r = kr
      rlabels = get_labels_from_Z(H.Z, r)
      nocs, noc_covers = H.get_ocn_discovery(k, rlabels)
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
      plot_h.core_dendrogram(
        [r], on=T, score="_"+score
      )
      plot_h.heatmap_pure(
        r, on=T, labels = rlabels,
        score=f"{r}_{nmi:.4f}"
      )
      plot_h.lcmap_dendro(
        [k], on=T, score="_"+score
      )
  print("End!")