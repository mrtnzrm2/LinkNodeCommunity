# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cut_tree
sns.set_theme()
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject = "MAC"
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
version = "57"+"d"+"106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = STR[f"{subject}{__inj__}"](
      linkage, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      discovery = discovery,
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = alpha
    )

    NET_H = read_class(
      NET.pickle_path,
      "hanalysis"
    )

    HeqZ = NET_H.equivalence
    H = NET_H.H
    Z = NET_H.Z

    nodes = NET.nodes
    edges = NET_H.leaves
    rows = NET.rows

    tree = {}
    tree_edges = {}
    count = nodes-1
    ecount = edges-1

    def insert_node(tree, key, node, c, partition, level):
      if isinstance(tree, dict):
        for k in tree.keys():
          if tree[k][0] == node:
            if key[0] < nodes:
              tree[k][-1][key] = [c+1, partition[key[0]], level, {}]
            else:
              tree[k][-1][key] = [c+1, -1, level, {}]
            return 
          else:
            insert_node(tree[k][-1], key, node, c, partition, level)


    # print([f"{i} {j}" for i, j in zip(cut_tree(Z, n_clusters=54).ravel(), cut_tree(Z, n_clusters=52).ravel()) if i != j])
    # b = {i: v for i, v in enumerate(cut_tree(Z, n_clusters=52).ravel())}
    # print(a)
    # print(b)

    # k_degeneracy = Counter(HeqZ[:, 0])
    # k_degeneracy = [k for k, v in k_degeneracy.items() if v > 1]
    
    # r_degeneracy = {}
    # for k in k_degeneracy:
    #   k_place = np.where(HeqZ[:, 0] == k)[0]
    #   for kp in k_place:
    #     r_degeneracy[HeqZ[kp, 1]] = k

    # for i in np.arange(edges-1):
    #   K = edges - 1 - i
    #   zmin = int(np.min([H[i, 0], H[i, 1]]))
    #   zmax = int(np.max([H[i, 0], H[i, 1]]))
    #   if zmin <= edges and zmax <= edges:
    #     tree_edges[(zmin, zmax)] = [ecount+1, {}, K]
    #   else:
    #     insert_node(tree_edges, (zmin, zmax), zmax, ecount)

    #   ecount += 1

    for i in np.arange(nodes-1):

      Level = nodes - 1 - i
      
      partition = cut_tree(Z, n_clusters=Level).ravel()

      zmin = int(np.min([Z[i, 0], Z[i, 1]]))
      zmax = int(np.max([Z[i, 0], Z[i, 1]]))
      if zmin < nodes and zmax < nodes:
        tree[(zmin, zmax)] = [count+1, partition[zmin], Level, {}]
      else:
        insert_node(tree, (zmin, zmax), zmax, count, partition, Level)

      count += 1

    
      
      
    