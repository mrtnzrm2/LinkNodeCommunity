# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
from various.omega import Omega
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
version = 103
mode = "ALPHA"
topology = "MIX"
index = "jacp"
# opt_score = ["_maxmu", "_X", "_D"]
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

gt_covers = {
  0 : ["A", "B", "C", "D"],
  1 : ["A", "E", "F", "G"]
}

def make_toys(linkage, **kwargs):
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
      TOY(toy_copy.astype(float), linkage, **kwargs)
    )
  return n, NETS

if __name__ == "__main__":
  N, NETS = make_toys(linkage, **properties)
  original_labels = np.array(["A", "B", "C", "D", "E", "F", "G"])
  labels_dict = dict()
  for i in np.arange(__nodes__):
    labels_dict[i] = original_labels[i]
  toy_names = np.arange(N).astype(int).astype(str)
  
  for i, net in enumerate(NETS):
    net.set_labels(
      original_labels
    )
    leaves = np.sum(net.A != 0)
    H = Hierarchy(
      net, net.A, net.A, np.zeros(net.A.shape),
      __nodes__, linkage, mode
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(net, labels=net.labels)
    L.get_regions()
    H.set_colregion(L)
    # Plot H ----
    plot_h = Plot_H(net, H)
    ## Omega ---
    for j, score in enumerate(opt_score):
      k, r = get_best_kr(score, H)
      rlabels = get_labels_from_Z(H.Z, r)
      _, nocs = H.get_ocn_discovery(k, rlabels)
      noc_covers = omega_index_format(rlabels, nocs, H.colregion.labels)
      omega = Omega(noc_covers, gt_covers).omega_score
      print(f"Omega index: {omega:.4f}")
      plot_h.lcmap_dendro(
        [k], cmap_name="deep",
        font_size=30,
        score="_"+score+"_"+toy_names[i], on=F
      )
      plot_h.plot_networx(
        r, rlabels, score="_"+score+"_"+toy_names[i],
        on=T, labels=labels_dict, cmap_name="deep"
      )
      plot_h.plot_networx_link_communities(
        [k], score="_"+score+"_"+toy_names[i],
        cmap_name="deep",
        on=F, labels=labels_dict
      )
      plot_h.core_dendrogram(
        [r], score="_"+score+"_"+toy_names[i],
        on=F, cmap_name="deep"
      )
