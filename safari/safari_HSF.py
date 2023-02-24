# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from various.network_tools import get_best_kr_equivalence, get_labels_from_Z, adj2df, match, get_best_kr

def append_letter(list_, letter):
  list_ = [f"{s}_{letter}" for s in list_]
  return list_

A = np.array(
  [
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0]
  ], dtype=float
)
seed = adj2df(A)
seed = seed.loc[seed.source < seed.target]
seed = seed.loc[seed.weight != 0]
seed.source = seed.source.to_numpy().astype(str)
seed.target = seed.target.to_numpy().astype(str)
seed.source = append_letter(seed.source, "00")
seed.target = append_letter(seed.target, "00")
seed.source[seed.source == "0_00"] = "A_00"
seed.target[seed.target == "0_00"] = "A_00"


class HSF:
  def __init__(self, edgelist, central_node, N : int, L : int) -> None:
    self.edgelist = edgelist
    self.central = central_node
    self.N = N
    self.L = L
    self.edgelist = self.H_edgelist()
    self.A_matrix()

  def H_edgelist(self):
    EDGELIST = self.edgelist.copy()
    for l in np.arange(1, self.L + 1):
      ED = pd.DataFrame()
      for n in np.arange(self.N):
        ed = EDGELIST.copy()
        ed.source = append_letter(ed.source, f"{l}{n}")
        ed.target = append_letter(ed.target, f"{l}{n}")
        ED = pd.concat([ED, ed], ignore_index=T)
      ED = pd.concat([ED, self.add_edges2center(ED)], ignore_index=T)
      EDGELIST = pd.concat([ED, EDGELIST], ignore_index=T)
    return EDGELIST

  def A_matrix(self):
    nodes = np.unique(list(self.edgelist.source) + list(self.edgelist.target))
    nodes = np.sort(nodes)
    self.edgelist.source = match(self.edgelist.source.to_numpy(), nodes)
    self.edgelist.target = match(self.edgelist.target.to_numpy(), nodes)
    self.nodes = len(nodes)
    self.labels = nodes
    self.A = np.zeros((self.nodes, self.nodes))
    self.A[self.edgelist.source, self.edgelist.target] = 1
    self.A = self.A + self.A.T   
  
  def add_edges2center(self, ed : pd.DataFrame):
    added_nodes = np.unique(
      list(ed.source) + list(ed.target)
    )
    added_nodes = [n for n in added_nodes if self.central not in n]
    toCenter = pd.DataFrame(
      {
        "source" : added_nodes,
        "target" : [self.central] * len(added_nodes),
        "weight" : [1] * len(added_nodes)
      }
    )
    return toCenter

if __name__ == "__main__":
  toy = HSF(seed, "A_00", 4, 2)
  perm = np.random.permutation(np.arange(toy.nodes))
  toy.A = toy.A[perm, :][:, perm]
  toy.labels = toy.labels[perm]

  linkage = "single"
  nlog10 = F
  lookup = F
  prob = F
  cut = T
  mode = "ALPHA"
  topology = "MIX"
  mapping="trivial"
  index = "tanimoto"
  opt_score = ["_maxmu", "_X", "_D"]

  properties = {
    "version" : "HSF",
    "nlog10" : nlog10,
    "lookup" : lookup,
    "prob" : prob,
    "cut" : cut,
    "topology" : topology,
    "mapping" : mapping,
    "index" : index,
    "mode" : mode,
  }

  NET = TOY(toy.A, linkage, **properties)
  NET.set_alpha([15, 30])
  NET.create_plot_directory()
  NET.set_labels(toy.labels)
  labels_dict = dict()
  for i in np.arange(toy.nodes): labels_dict[i] = toy.labels[i]
  H = Hierarchy(
    NET, toy.A, toy.A, np.zeros(toy.A.shape),
    toy.nodes, linkage, mode
  )
  ## Compute topologys ----
  H.BH_features_cpp()
  ## Compute la arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  # Set labels to network ----
  L = colregion(NET, labels=NET.labels)
  L.get_regions()
  H.set_colregion(L)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_h.plot_measurements_D(on=T)
  plot_h.plot_measurements_X(on=T)
  plot_h.plot_measurements_mu(on=T)
  for score in opt_score:
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    _, nocs_membership = H.get_ocn_discovery(k, rlabels)
    print(nocs_membership)
    plot_h.lcmap_dendro(
      [k], cmap_name="husl",
      font_size=7, remove_labels=F,
      score="_"+score, on=T
    )
    plot_h.core_dendrogram(
      [r], score="_"+score,
      cmap_name="husl", remove_labels=F, on=T
    )
