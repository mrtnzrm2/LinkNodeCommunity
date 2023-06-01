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
from modules.hierarentropy import Hierarchical_Entropy
from various.network_tools import *

def worker_ER(
  number_of_iterations : int, number_of_nodes : int, rho : float,
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
    "version" : "ER",
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
  M = int(N * (N - 1) * rho)
  labels = np.arange(N).astype(int).astype(str)
  data = TOYH()
  for i in np.arange(MAXI):
    data.set_iter(i)
    G = nx.gnm_random_graph(N, M, directed=T)
    A = nx.adjacency_matrix(G).todense()
    A = np.array(A, dtype=float)
    # Create TOY ---
    NET = TOY(A, linkage, **properties)
    NET.set_alpha([6, 15, 30])
    NET.create_plot_directory()
    NET.set_labels(labels)
    H = Hierarchy(
      NET, A, A, np.zeros(A.shape),
      N, linkage, mode
    )
    ## Compute features ----
    H.BH_features_parallel()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Entropy ----
    HS = Hierarchical_Entropy(H.Z, H.nodes, labels)
    HS.Z2dict("short")
    HS.zdict2newick(HS.tree, weighted=F, on=F)
    HS.zdict2newick(HS.tree, weighted=T, on=F)
    node_entropy = HS.S(HS.tree)
    node_entropy_H = HS.S_height(HS.tree)
    data.update_entropy(
      [node_entropy, node_entropy_H, H.link_entropy, H.link_entropy_H],  
    )
  data.set_subfolder(H.subfolder)
  data.set_pickle_path(H, bias=bias)
  print("Save data")
  save_class(data, data.pickle_path, f"series_{MAXI}_{rho:.2f}_{number_of_nodes}")
  print("End")
