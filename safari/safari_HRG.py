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
from various.network_tools import get_best_kr, get_labels_from_Z
from various.data_transformations import maps

class HRG:
  def __init__(self, N : int, group_sizes=[160, 40, 10], rho=1, kav=16, seed=-1) -> None:
    self.nodes = N
    self.rho = rho
    self.kav = kav
    self.group_sizes = np.array(group_sizes)
    self.lmax = len(group_sizes)
    self.A = np.zeros((N, N))
    self.labels = np.arange(N, dtype=int).astype(str)
    self.generate_membership_list()
    self.fill_adjacency_matrix(seed=seed)

  def get_p(self):
    p = np.zeros(len(self.group_sizes) + 1)
    p[0] = np.power(self.rho, self.lmax) / np.power(1 + self.rho, self.lmax) * (self.kav / (self.group_sizes[0] * self.lmax))
    for x, Sx in enumerate(self.group_sizes):
      x += 1
      p[x] = np.power(self.rho, self.lmax - x) / np.power(1 + self.rho, self.lmax - x + 1) * (self.kav / (Sx - 1))
    return p

  def fill_adjacency_matrix(self, seed=-1):
    if seed > 0:
      np.random.seed(seed)
    p = self.get_p()
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
    memberships_options = np.arange(self.lmax + 1)
    self.node_memberships = np.zeros((self.nodes, len(self.group_sizes)))
    for i, size in enumerate(self.group_sizes):
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
index = "bsim"
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
  # Define parameters HRG
  N = 160
  group_sizes = [40, 20, 5]
  kav = 7
  #
  hrg = HRG(N, group_sizes=group_sizes, kav=kav, seed=12345)
  perm = np.random.permutation(np.arange(N))
  hrg.A = hrg.A[perm, :][:, perm]
  labels_dict = {i: hrg.labels[i] for i in np.arange(N)}
  # Create TOY ---
  NET = TOY(hrg.A, linkage, **properties)
  NET.set_alpha([6, 15, 30])
  NET.create_plot_directory()
  NET.set_labels(hrg.labels)
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=0
  )
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
    K, R = get_best_kr(score, H)
    r = R[K == np.min(K)][0]
    k = K[K == np.min(K)][0]
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
