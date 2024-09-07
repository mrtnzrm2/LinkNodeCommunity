# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Personal libs ---- 
from networks.MAC.mac57 import MAC57
from networks.structure import STR
from modules.hierarentropy import Hierarchical_Entropy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.0
opt_score = ["_S"]
__nodes__ = 40
__inj__ = 40
total_nodes = 91
version = f"{__nodes__}d{total_nodes}"
save_data = T
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = STR[f"{subject}{__nodes__}"](
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )
  
  H = read_class(
    NET.pickle_path,
    "hanalysis"
  )

  tree = formating_Z2HMI(H.Z, H.nodes)
  from various.hit import check, EHMI, HMI
  print(check(tree))
  print(tree)

  # tr1 = [[[0], [[1, 2], [[3, 4, 5]]]]]
  # tr2 = [[[0], [[1, 2], [3, 4, 5]]]]
  # tr3 = [[[0], [[1, 2], [[3], [4, 5]]]]]

  # tr1 = [[0], [[1, 2, 3]]]
  # tr2 = [[3], [1, 0, 2]]


  # print(HMI(tr1, tr2))
  # print(EHMI(tr1, tr3))
  print(EHMI(tree, tree))

  # labels = NET.struct_labels

  # def replace_tree_label(tree, id, label):
  #   if isinstance(tree, list):
  #     for i, t in enumerate(tree):
  #       if t == id: tree[i] = label
  #       else: replace_tree_label(t, id, label)

  # for i in range(NET.nodes):
  #   l = labels[i]
  #   replace_tree_label(tree, i, l)

  # print(tree)



  # for i, val in tree.items():
  #   print(i)

  # print(list(tree["L0_0"].keys()))

  # tree = {"L0_0": {"L1_0":1, "L2_0":1}, "L1_0" : {0 : 1, 1:1}, "L2_0" : {"L1_1" : 1, "L2_1":1}, "L1_1" : {2:1, 3:1}, "L2_1":{"L3_1": 1}, "L3_1" : {4:1, 5:1}}
  
  # # print(tree)
  # print(tree)
  # tree_hp = []

  # get_levels = np.unique([int(f.split("_")[-2].split("L")[-1]) for f in tree.keys()]).shape[0]

  # for i in np.arange(get_levels-1):
  #   tree_hp = [tree_hp]

  # def crazy(tree : dict, root, tree2 : list):
  #   tree_keys = list(tree[root].keys())
  #   tree_true_keys = [tr for tr in tree_keys if tr in list(tree.keys())]
  #   tree_false_keys = [tr for tr in tree_keys if tr not in list(tree.keys())]
  #   for tr in tree_true_keys:
  #     if tr in tree.keys():
  #         crazy(tree, tr, tree2[0])
  #   else:
  #     if (len(tree_false_keys) > 0):
  #       return tree2.append(tree_false_keys)
      
  # crazy(tree, "L0_0", tree_hp)
  # print(tree_hp)

  # from various.hit import check

  # print(check(tree_hp[0]))
