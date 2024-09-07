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
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Personal libs ---- 
from networks.toy import TOY
from networks.MAC.mac57 import MAC57
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.hierarentropy import Hierarchical_Entropy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.data_transformations import maps
from modules.discovery import discovery_channel
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "57d106"
__nodes__ = 57
__inj__ = 57

NET = MAC57(
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

A = NET.A
CC = NET.CC
D = NET.D

permc = np.random.permutation(np.arange(NET.nodes))
permr = np.random.permutation(np.arange(NET.nodes, NET.rows))
permr = np.hstack([permc, permr])

A = A[permr, :][:, permc]
D = D[permr, :][:, permr]
CC = CC[permr, :][:, permc]

RAND = MAC57(
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

RAND.A = A
RAND.D = D
RAND.CC = CC
RAND.struct_labels = RAND.struct_labels[permr]

H = Hierarchy(
  RAND, A, A, D,
  __nodes__, linkage, mode, lookup=lookup,
  index=index
)
## Compute features ----
H.BH_features_cpp_no_mu()
## Compute lq arbre de merde ----
H.la_abre_a_merde_cpp(H.BH[0])
# Set labels to network ----
L = colregion(RAND, labels_name=f"labels{__inj__}")
H.set_colregion(L)
H.colregion.labels = RAND.struct_labels
H.delete_dist_matrix()

RN = RAND.A[:__nodes__, :].copy()
RN[RN > 0] = -np.log(RN[RN > 0])
np.fill_diagonal(RN, 0.)

# Picasso ----
plot_h = Plot_H(NET, H)
plot_n = Plot_N(NET, H)

for SCORE in opt_score:
    # Get best K and R ----
    K, R, TH = get_best_kr_equivalence(SCORE, H)
    for k, r, th in zip(K, R, TH):
      print(f"Find node partition using {SCORE}")
      print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, SCORE))
      H.set_kr(k, r, score=SCORE)
      rlabels = get_labels_from_Z(H.Z, r)
      rlabels = skim_partition(rlabels)

      print(">>> Single community nodes:")
      print(NET.struct_labels[:NET.nodes][np.where(rlabels == -1)[0]], "\n")

      # plot_h.heatmap_dendro(r, -RN, on=F, score="FLNe", cbar_label=r"$\log_{10}$FLNe", font_size = 12, suffix="")
      
      # Overlap ----
      for direction in ["both"]: # ,  "source", "target",
        print("***", direction)
        RAND.overlap, RAND.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)
        print(">>> Areas with predicted overlapping communities:\n",  RAND.data_nocs, "\n")
        cover = omega_index_format(rlabels2,  RAND.data_nocs, RAND.struct_labels[:RAND.nodes])

        # while np.sum(rlabels2 == -1) > 0:
        #   RAND.overlap, RAND.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels2, direction=direction, index=index)
        #   print(">>> Areas with predicted overlapping communities:\n",  RAND.data_nocs, "\n")
        #   cover = omega_index_format(rlabels2,  RAND.data_nocs, RAND.struct_labels[:RAND.nodes])

        # print(rlabels2)

        plot_n.plot_network_covers(
          k, RN, rlabels2, rlabels,
          RAND.data_nocs,
          # cover_art,
          noc_sizes, RAND.struct_labels[:RAND.nodes], ang=0,
          # color_order=color_order,
          score=SCORE, direction=direction, spring=F, font_size=16,
          scale=0.45,
          suffix="small", cmap_name="hls", not_labels=F, on=T#, figsize=(8,8)
        )