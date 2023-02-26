# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
version = 105
mode = "ALPHA"
topology = "SOURCE"
index = "cos"
# opt_score = ["_maxmu", "_D"]
opt_score = ["_maxmu"]
save_data = T
__nodes__ = 7

properties = {
  "version" : version,
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "topology" : topology,
  "index" : index,
  "mode" : mode
}

ww = 0
ww2 = 0
wm = 60
ws = 100
toy = np.array(
  [
    [0, wm, wm, wm, wm, wm, wm],
    [wm, 0, ws, ws, ww, ww, ww],
    [wm, ws, 0, ws, ww, ww, ww],
    [wm ,ws, ws, 0, ww, ww, ww],
    [wm, ww2, ww2, ww2, 0, ws, ws],
    [wm, ww2, ww2, ww2, ws, 0, ws],
    [wm, ww2, ww2, ww2, ws, ws, 0]
  ]
)

# toy = np.array(
#   [
#     [0, ww, ww, ww, ww, ww, ww],
#     [ww, 0, ws, ws, wm, wm, wm],
#     [ww, ws, 0, ws, wm, wm, wm],
#     [ww ,ws, ws, 0, wm, wm, wm],
#     [ww, wm, wm, wm, 0, ws, ws],
#     [ww, wm, wm, wm, ws, 0, ws],
#     [ww, wm, wm, wm, ws, ws, 0]
#   ]
# )


if __name__ == "__main__":
  NET = TOY(toy, linkage, **properties)
  original_labels = np.array(["A", "B", "C", "D", "E", "F", "G"])
  labels_dict = dict()
  for i in np.arange(__nodes__):
    labels_dict[i] = original_labels[i]
  NET.set_labels(
    original_labels
  )
  H = Hierarchy(
    NET, NET.A, NET.A, np.zeros(NET.A.shape),
    __nodes__, linkage, mode
  )
  ## Compute features ----
  H.BH_features_cpp()
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  # Set labels to network ----
  L = colregion(NET, labels=NET.labels)
  L.get_regions()
  H.set_colregion(L)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  for j, score in enumerate(opt_score):
    k, r = get_best_kr(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    _, nocs_membership = H.get_ocn_discovery(k, rlabels)
    print(nocs_membership)
    plot_h.lcmap_dendro(
      [k], cmap_name="deep",
      font_size=30, score="_"+score, on=T
    )
    plot_h.plot_networx(
      r, rlabels, score="_"+score,
      on=T, labels=labels_dict, cmap_name="deep"
    )
    plot_h.plot_networx_link_communities(
      [k], score="_"+score, cmap_name="deep",
      on=T, labels=labels_dict
    )
    plot_h.core_dendrogram(
      [r], score="_"+score, on=T, cmap_name="deep"
    )
