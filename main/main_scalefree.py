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
from networks.scalefree import SCALEFREE
from modules.colregion import colregion
from numpy import zeros
from various.network_tools import *

__iter__ = 0
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
topology = "SOURCE"
mapping = "trivial"
index  = "jacp"
__mode__ = "ALPHA"
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
# WDN paramters ----
par = {
  "-N" : "{}".format(str(__nodes__)),
  "-k" : "25.0",
  "-maxk" : "100",
  "-mut" : "0.2",
  "-muw" : "0.2",
  "-beta" : "2.5",
  "-t1" : "2.5",
  "-t2" : "2.5"
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
    H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET)
    H.set_colregion(L)
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(NET.subfolder),
    )
  # Plot H ----
  plot_h = Plot_H(NET, H, sln=F)
  plot_h.Mu_plotly(on=T)
  plot_h.D_plotly(on=T)
  plot_h.X_plotly(on=T)
  # Plot N ----
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(NET.A, s=5, on=F)
  plot_n.histogram_weight(on=T)
  plot_n.plot_aki(s=1, on=F)
  plot_h.heatmap_pure(
    0, name = "_GT_{}".format(number_of_communities),
    labels = NET.labels, on=T
  )
  # Find best k partition ----
  for score in opt_score:
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    # Check labels safety ----
    if np.nan in rlabels:
        print("Warning: Impossible node dendrogram")
        break
    #Prints ----
    nmi = AD_NMI_label(
      NET.labels, rlabels,
      on=T
    )
    plot_h.core_dendrogram([r], on=F)
    ## Single linkage ----
    plot_h.heatmap_pure(
      r, on=T, labels = rlabels, name=f"{r}_{nmi:.4f}"
    )
    plot_h.heatmap_dendro([k], on=F)
    plot_h.lcmap_dendro(
      [k], score="_"+score, on=T
    )
    plot_h.lcmap_pure(
      [r],
      labels=rlabels,
      on = F
    )
  print("End!")
  # #@@ Todo: