# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
T = True
F = False
# Stadard python libs ----
import numpy as np
import networkx as nx
# Personal libs ----
from networks.toy import TOY
from networks_serial.toyh import TOYH
from modules.hierarmerge import Hierarchy
from various.network_tools import *

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

def worker_HRG(
  number_of_iterations : int, number_of_nodes : int,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool,
  topology : str, mapping : str, index : str, bias : float, mode : str
):
  # Declare global variables ----
  linkage = "single"
  nlog10 = F
  lookup = F
  prob = F
  cut = T
  mode = mode
  topology = topology
  mapping = mapping
  index = index

  properties = {
    "version" : "HRG",
    "nlog10" : nlog10,
    "lookup" : lookup,
    "prob" : prob,
    "cut" : cut,
    "mapping" : mapping,
    "topology" : topology,
    "index" : index,
    "mode" : mode
  }
  MAXI = number_of_iterations
  N = number_of_nodes
  labels = np.arange(N).astype(int).astype(str)
  data = TOYH()
  for i in np.arange(MAXI):
    data.set_iter(i)
    hrg = HRG(N)
    perm = np.random.permutation(np.arange(N))
    hrg.A = hrg.A[perm, :][:, perm]
    # Create TOY ---
    NET = TOY(hrg.A, linkage, **properties)
    NET.set_alpha([6, 15, 30])
    NET.create_plot_directory()
    NET.set_labels(labels)
    H = Hierarchy(
      NET, hrg.A, hrg.A, np.zeros(hrg.A.shape),
      N, linkage, mode
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    # Update wntropy ----
    data.update_entropy(
      [H.node_entropy, H.node_entropy_H, H.link_entropy, H.link_entropy_H],  
    )
  data.set_subfolder(H.subfolder)
  data.set_pickle_path(H, bias=bias)
  print("Save data")
  save_class(data, data.pickle_path, f"series_{MAXI}")
  print("End")
