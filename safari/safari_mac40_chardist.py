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
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
# Personal libs ---- 
from networks.MAC.mac40 import MAC40
from modules.discovery import discovery_channel
from various.data_transformations import maps
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "40d91"
__nodes__ = 40
__inj__ = 40

def get_cover(NET):
  R, _, _ = maps[mapping](
    NET.A, F, T, T, b=0.0
  )

  H = Hierarchy(
    NET, NET.A, R, NET.D,
    __nodes__, linkage, mode, lookup=F,
    index="Hellinger2"
  )

  ## Compute features ----
  H.BH_features_cpp_no_mu()
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  H.get_h21merge()
  # Set labels to network ----
  L = colregion(NET, labels_name=f"labels{__inj__}")
  H.set_colregion(L)
  # Save ----
  H.delete_dist_matrix()

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

      # Overlap ----
      for direction in ["both"]:
        print("***", direction)
        NET.overlap, NET.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)
        print(">>> Areas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
        cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])

    return cover
  
# Start main ----9
if __name__ == "__main__":
  # Load structure ----
  NET = MAC40(
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

  NETRED = MAC40(
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

  dav = 0.5 * (13.50897609007985 + 12.459343635601233)

  omit = NETRED.D[:, :NETRED.nodes] < dav

  NETRED.CC[~omit] = 0
  NETRED.A = NETRED.CC / np.sum(NETRED.CC, axis=0)

  coverNormal = get_cover(NET)
  coverRed = get_cover(NETRED)

  omega_index(coverNormal, coverRed)

  

