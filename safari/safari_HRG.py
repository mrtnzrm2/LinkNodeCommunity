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
# Personal libs ----
from networks.toy import TOY
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from plotting_modules.plotting_H import Plot_H
from various.network_tools import get_best_kr_equivalence, get_labels_from_Z

class HRG:
  def __init__(self, N : int, seed=-1) -> None:
    self.nodes = N
    self.A = np.zeros((N, N))
    self.labels = np.arange(N, dtype=int).astype(str)
    self.generate_membership_list()
    self.fill_adjacency_matrix(seed=seed)

  def fill_adjacency_matrix(self, seed=-1):
    if seed > 0:
      np.random.seed(seed)
    p = np.array([
      2/360, 2/159, 4/39, 8/9
    ])
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if i == j: continue
        members_i = self.node_memberships[i, :]
        members_j = self.node_memberships[j, :]
        compare = members_i == members_j
        kk = 0
        for k in np.arange(self.node_memberships.shape[1]):
          if compare[k]: kk += 1
          else: break
        if np.random.rand() < p[kk]: self.A[i, j] = 1      

  def generate_membership_list(self):
    group_sizes = np.array([160, 40, 10])
    memberships_options = np.arange(4)
    self.node_memberships = np.zeros((self.nodes, len(group_sizes)))
    for i, size in enumerate(group_sizes):
      mem = np.repeat(memberships_options, size)
      if len(mem) < self.nodes:
        mem = np.tile(mem, int(self.nodes / len(mem)))
      self.node_memberships[:, i] = mem

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = T
mode = "ALPHA"
topology = "MIX"
mapping="trivial"
index = "jacp"
opt_score = ["_maxmu", "_X", "_D"]

properties = {
  "version" : "HRG",
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "topology" : topology,
  "mapping" : mapping,
  "index" : index,
  "mode" : mode,
}

if __name__ == "__main__":
  N = 640
  hrg = HRG(N, seed=12345)
  perm = np.random.permutation(np.arange(N))
  hrg.A = hrg.A[perm, :][:, perm]
  labels_dict = dict()
  for i in np.arange(N):
    labels_dict[i] = hrg.labels[i]
  # Create TOY ---
  NET = TOY(hrg.A, linkage, **properties)
  NET.create_plot_directory()
  NET.set_labels(hrg.labels)
  H = Hierarchy(
    NET, hrg.A, hrg.A, np.zeros(hrg.A.shape),
    N, linkage, mode
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
  for j, score in enumerate(opt_score):
    k, r = get_best_kr_equivalence(score, H)
    rlabels = get_labels_from_Z(H.Z, r)
    NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    plot_h.lcmap_dendro(
      [k], cmap_name="husl",
      font_size=30, remove_labels=T,
      score="_"+score, on=T
    )
    plot_h.core_dendrogram(
      [r], score="_"+score,
      cmap_name="husl", remove_labels=T, on=T
    )
