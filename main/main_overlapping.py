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
from plotting_modules.plotting_N import Plot_N
from plotting_modules.plotting_overlap import PLOT_O
from networks.overlapping import OVERLAPPING
from modules.colregion import colregion
from numpy import zeros
from various.network_tools import *
# Declare global variables ----
__iter__ = 0
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = T
run = T
topology = "SOURCE"
mapping = "trivial"
index  = "cos"
__mode__ = "ALPHA"
opt_score = ["_maxmu", "_X", "_D"]
save_datas = T
# Overlapping WDN paramters ----
opar = {
  "-N" : "{}".format(
    str(__nodes__)
  ),
  "-k" : "25.0",
  "-maxk" : "100",
  "-mut" : "0.2",
  "-muw" : "0.4",
  "-beta" : "2.5",
  "-t1" : "2.5",
  "-t2" : "2.5",
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
      __nodes__, linkage, __mode__
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    H.set_colregion(L)
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(NET.subfolder)
    )
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_h.Mu_plotly(on=T)
  plot_h.D_plotly(on=T)
  plot_h.X_plotly(on=T)
  # Plot O ----
  plot_o = PLOT_O(NET, H)
  # Plot N ----
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(NET.A, s=5, on=F)
  plot_n.histogram_weight(on=F)
  plot_n.plot_aki(s=1, on=F)
  for score in opt_score:
    # Find best k partition ----
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    #Prints ----
    nmi = AD_NMI_overlap(
      NET.labels, rlabels, NET.overlap, on=T
    )
    plot_h.heatmap_pure(
      r, name = "_GT",
      labels = NET.labels, on=T
    )
    ##
    # sen, sep = NET.overlap_score(
    #   H, [k], rlabels, on=T
    # )
    sen, sep = NET.overlap_score_discovery( H, k, rlabels, on=T)
    ##
    plot_h.core_dendrogram(
      [r], on=T, score="_"+score
    )
    ## Single linkage ----
    plot_h.heatmap_pure(
      r, on=T, labels = rlabels,
      score=f"{r}_{nmi:.4f}"
    )
    plot_h.heatmap_dendro(
      r, on=T, score="_"+score
    )
    plot_h.lcmap_dendro(
      [k], on=T, score="_"+score
    )
    plot_h.lcmap_pure(
      [k],
      labels = NET.labels,
      on = F
    )
    plot_o.bar_node_membership(
      [k], labels = rlabels, on=F
    )
    plot_o.bar_node_overlap(
      [k], NET.overlap,on=F
    )
  print("End!")