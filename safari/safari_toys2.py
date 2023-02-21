# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_overlap import PLOT_O
from various.network_tools import get_labels_from_Z, get_best_kr_equivalence, print_principal_memberships

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
version = 102
mode = "ALPHA"
feature = "MIX"
# opt_score = ["_maxmu", "_X", "_D"]
opt_score = ["_maxmu"]
save_data = T
__nodes__ = 7
__inj__ = 7

properties = {
  "version" : version,
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "feature" : feature,
  "mode" : mode,
  "inj" : __inj__
}

ww = 1
ws = 5

toy = np.array(
  [
    [0, -1, -1, -1, -1, -1, -1],
    [-1, 0, ws, ws, 0, 0, 0],
    [-1, 0, 0, ws, 0, 0, 0],
    [-1 ,ws ,0, 0, 0, 0 ,0],
    [-1, 0, 0, 0, 0, ws, ws],
    [-1, 0, 0, 0, ws, 0, 0],
    [-1, 0, 0, 0, ws, ws, 0]
  ]
)

n = (toy.shape[0] - 1) * 2

def make_toys():
  A = np.array(
    [
      ww, ww, ww,
      ws, ws, ws,
      0, 0, 0,
      0, 0, 0
    ]
  )
  x = np.arange(n)
  NETS = []
  rows, cols = np.where(toy == -1)
  for i in np.arange(n):
    wheel = np.zeros(n)
    wheel[i:] = x[:(n-i)]
    wheel[:i] = x[(n-i):]
    wheel = wheel.astype(int)
    toy_copy = toy.copy()
    toy_copy[rows, cols] = A[wheel]
    NETS.append(
      TOY(toy_copy.astype(float), linkage, **properties)
    )
  return NETS

if __name__ == "__main__":
  original_labels = np.array(["A", "B", "C", "D", "E", "F", "G"])
  labels_dict = dict()
  for i in np.arange(__nodes__):
    labels_dict[i] = original_labels[i]
  toy_names = np.arange(n).astype(int).astype(str)
  NETS = make_toys()
  A_ocn = np.zeros(
    (len(NETS), len(opt_score))
  )
  for i, net in enumerate(NETS):
    net.set_labels(
      original_labels
    )
    leaves = np.sum(net.A != 0)
    H = Hierarchy(
      net, net.A, net.A, np.zeros(net.A.shape),
      __nodes__, linkage, mode, prob=prob
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(net, labels=net.labels)
    L.get_regions()
    H.set_colregion(L)
    ##
    ocn_sack = np.zeros(len(opt_score))
    # Plot H ----
    plot_h = Plot_H(net, H)
    # Plot O ----
    plot_o = PLOT_O(net, H)
    for j, score in enumerate(opt_score):
      k, r = get_best_kr_equivalence(score, H)
      rlabels = get_labels_from_Z(H.Z, r)
      net.overlap, nocs_membership = H.get_ocn_discovery(k, rlabels)
      print(nocs_membership)
      print_principal_memberships(rlabels, H.colregion.labels)
      H.set_overlap_labels(net.overlap, score)
      plot_h.lcmap_dendro(
        [k], cmap_name="deep",
        font_size=30,
        score="_"+score+"_"+toy_names[i], on=T
      )
      plot_h.plot_networx(
        r, rlabels, score="_"+score+"_"+toy_names[i],
        on=T, labels=labels_dict, cmap_name="deep"
      )
      plot_h.plot_networx_link_communities(
        [k], score="_"+score+"_"+toy_names[i],
        cmap_name="deep",
        on=T, labels=labels_dict
      )
      plot_h.core_dendrogram(
        [r], score="_"+score+"_"+toy_names[i],
        on=T, cmap_name="deep"
      )
      plot_o.bar_node_membership(
        [k], labels = rlabels, score="_"+score+"_"+toy_names[i],
        node_labels = original_labels, on=T,
      )
      plot_o.bar_node_overlap(
        [k], net.overlap, score="_"+score+"_"+toy_names[i],
        node_labels = original_labels, on=T
      )
      if net.overlap[0] ==  "A":
        ocn_sack[j] = 1
    A_ocn[i, :] = ocn_sack
  #Create data ----
  A_ocn = pd.DataFrame(A_ocn, columns=opt_score)
  A_ocn["x"] = np.arange(A_ocn.shape[0])
  AOCN = pd.DataFrame()
  for i in np.arange(len(opt_score)):
    AOCN = pd.concat(
      [
        AOCN, 
        pd.DataFrame(
          {
            "x" : A_ocn.x.to_numpy().astype(str),
            "OCN" : A_ocn[opt_score[i]],
            "score" : [opt_score[i]] * len(NETS)
          }
        )
      ], ignore_index=T
    )
  # Create figure ----
  fig, ax = plt.subplots(1, 1)
  sns.lineplot(
    data=AOCN,
    x="x",
    y="OCN",
    hue="score",
    ax=ax
  )
  fig.tight_layout()
  # Crate path ----
  # Save plot ----
  plt.savefig(
    "A_OCN_x.png",
    dpi = 300
  )
  plt.close()
