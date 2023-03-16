# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stardard python libs ----
import numpy as np
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from plotting_modules.plotting_H import Plot_H
from scipy.cluster.hierarchy import cut_tree
from various.network_tools import get_best_kr, get_labels_from_Z, adj2df

# Create toy networks ----
ww = 1
ws = 5
toy_nor = np.array(
  [
    [0, 0, ws, ws, ww, ww, ww],
    [ws, 0, ws, ws, 0, 0, 0],
    [ws, 0, 0, ws, 0, 0, 0],
    [0, ws, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, ws, ws],
    [0, 0, 0, 0,  ws, 0, 0],
    [0, 0, 0, 0, ws, ws, 0]
  ]
)

toy_out = np.array(
  [
    [0, ws, ws, ws, ww, ww, ww],
    [0, 0, ws, ws, 0, 0, 0],
    [0, 0, 0, ws, 0 ,0, 0],
    [0, ws, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, ws, ws],
    [0, 0, 0, 0, ws, 0, 0],
    [0, 0, 0, 0, ws, ws, 0]
  ]
)

toy_in = np.array(
  [
    [0, 0, 0, 0, 0, 0, 0],
    [ws, 0, ws, ws, 0, 0, 0],
    [ws, 0, 0, ws, 0, 0, 0],
    [ws ,ws ,0, 0, 0, 0 ,0],
    [ww, 0, 0, 0, 0, ws, ws],
    [ww, 0, 0, 0, ws, 0, 0],
    [ww, 0, 0, 0, ws, ws, 0]
  ]
)

toy_flow = np.array(
  [
    [0, 0, 0, 0, ww, ww, ww],
    [ws, 0, ws, ws, 0, 0, 0],
    [ws, 0, 0, ws, 0, 0, 0],
    [ws, ws, 0, 0, 0, 0 ,0],
    [0, 0, 0, 0, 0, ws, ws],
    [0, 0, 0, 0, ws, 0, 0],
    [0, 0, 0, 0, ws, ws, 0]
  ]
)

if __name__ == "__main__":
  # Create the nodes' labels ----
  original_labels = np.array(["A", "B", "C", "D", "E", "F", "G"])
  labels_dict = dict()
  for i in np.arange(len(original_labels)):
    labels_dict[i] = original_labels[i]
  toy_names = [
    "normal",
    "in",
    "out",
    "flow"
  ]
  # Declare global variables ----
  linkage = "single"
  nlog10 = F
  lookup = F
  prob = F
  cut = F
  version = 100
  mode = "ALPHA"
  nature = "original"
  feature = "MIX"
  index = "jacp"
  mapping="trivial"
  opt_score = ["_D"]
  # opt_score = ["_maxmu"]
  nodes = len(original_labels)
  properties = {
    "version" : version,
    "nlog10" : nlog10,
    "lookup" : lookup,
    "prob" : prob,
    "cut" : cut,
    "feature" : feature,
    "mode" : mode
  }
  # List toys ----
  NETS = [
    TOY(toy_nor, linkage, **properties),
    TOY(toy_in, linkage, **properties),
    TOY(toy_out, linkage, **properties),
    TOY(toy_flow, linkage, **properties) 
  ]
  for i, net in enumerate(NETS):
    if i != 0: continue
    net.set_beta([0.01])
    net.set_labels(original_labels)
    leaves = np.sum(net.A != 0)
    H = Hierarchy(
      net, net.A, net.A, np.zeros(net.A.shape),
      nodes, linkage, mode
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Entropy ----
    HS = Hierarchical_Entropy(H.H, H.leaves, labels=list(range(H.leaves)))
    HS.Z2dict("short")
    HS.zdict2newick(HS.tree, weighted=F, on=F)
    HS.zdict2newick(HS.tree, weighted=T, on=F)
    node_entropy = HS.S(HS.tree)
    node_entropy_H = HS.S_height(HS.tree)
    H.entropy = [
      node_entropy, node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    # Set labels to network ----
    L = colregion(net, labels=net.labels)
    L.get_regions()
    H.set_colregion(L)
    plot_h = Plot_H(net, H)
    plot_h.plot_measurements_Entropy(on=T)
    for score in opt_score:
      k, r = get_best_kr(score, H)
      rlabels = get_labels_from_Z(H.Z, r)
      net.overlap, noc_covers = H.get_ocn_discovery(k, rlabels)
      H.set_overlap_labels(net.overlap, score)
      plot_h.lcmap_dendro(
        [k], cmap_name="deep",
        font_size=30,
        score="_"+score+"_"+toy_names[i], on=F
      )
      plot_h.plot_networx(
        r, rlabels, score="_"+score+"_"+toy_names[i],
        on=F, labels=labels_dict, cmap_name="deep"
      )
      plot_h.plot_networx_link_communities(
        [k], score="_"+score+"_"+toy_names[i],
        cmap_name="deep",
        on=F, labels=labels_dict
      )
      plot_h.core_dendrogram(
        [r], score="_"+score+"_"+toy_names[i],
        on= T, cmap_name="deep"
      )