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
from modules.hierarentropy import Hierarchical_Entropy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from networks.scalefree import SCALEFREE
from modules.colregion import colregion
from numpy import zeros
from various.network_tools import *

__iter__ = 0
__nodes__ = 200
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
run = T
topology = "SOURCE"
mapping = "trivial"
index  = "simple2"
__mode__ = "ALPHA"
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
# WDN paramters ----
par = {
  "-N" : "{}".format(str(__nodes__)),
  "-k" : "10",
  "-maxk" : "50",
  "-mut" : "0.1",
  "-muw" : "0.3",
  "-beta" : "3",
  "-t1" : "2",
  "-t2" : "1",
  "-nmin" : "10",
  "-nmax" : "20"
}
if __name__ == "__main__":
  # Create EDR network ----
  NET = SCALEFREE(
    __iter__,
    linkage,
    __mode__,
    nlog10=nlog10,
    lookup=lookup,
    cut=cut,
    topology=topology,
    mapping=mapping, index=index,
    parameters = par
  )
  NET.create_plot_path()
  NET.create_pickle_path()
  NET.set_alpha([6, 50, 100])
  NET.set_beta([0.1, 0.2, 0.4])
  # Create network ----
  print("Create random graph")
  NET.random_WDN_cpp(run=run, on_save_pickle=T)
  NET.col_normalized_adj(on=F)
  number_of_communities = len(np.unique(NET.labels))
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, NET.A, zeros(NET.A.shape),
      __nodes__, linkage, __mode__
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
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=T
    )
    HS = Hierarchical_Entropy(H.Z, H.nodes, list(range(H.nodes)))
    HS.Z2dict("short")
    node_entropy = HS.S(HS.tree)
    node_entropy_H = HS.S_height(HS.tree)
    H.entropy = [
      node_entropy, node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=T
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(NET.subfolder),
    )
  # Plot H ----
  plot_h = Plot_H(NET, H, sln=F)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_mu(on=T)
  plot_h.plot_measurements_X(on=T)
  plot_h.heatmap_pure(
    0, np.log(1 + NET.A), score = "_GT_{}".format(number_of_communities),
    labels = NET.labels, on=T
  )
  # Find best k partition ----
  for score in opt_score:
    K, R = get_best_kr(score, H)
    for ii, kr in enumerate(zip(K, R)):
      k, r = kr
      rlabels = get_labels_from_Z(H.Z, r)
      # Check labels safety ----
      if np.nan in rlabels:
          print("Warning: Impossible node dendrogram")
          break
      ## Prints ----
      nmi = AD_NMI_label(NET.labels, rlabels, on=T)
      ## Plots ----
      plot_h.core_dendrogram([r], on=F)
      plot_h.heatmap_pure(
        r, np.log(1+NET.A), on=T, labels = rlabels, name=f"{r}_{nmi:.4f}"
      )
      plot_h.heatmap_dendro(r, np.log(1+NET.A), on=F)
      plot_h.lcmap_dendro(
        k, np.log(1+NET.A), score="_"+score, on=T
      )
      plot_h.lcmap_pure(
        [r],
        labels=rlabels,
        on = F
      )
  print("End!")
  # #@@ Todo: