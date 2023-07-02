# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from modules.colregion import colECoG
from various.data_transformations import maps
from networks.ECoG.structure import WAVES
from various.network_tools import *
# Iterable varaibles ----
def worker_ECoG(
    nature : str, mode : str, topology : str, mapping :str,
    index : str, nlog10 : bool, lookup : bool, cut : bool
):
  # Declare global variables ----
  linkage = "single"
  nlog10 = nlog10
  lookup = lookup
  prob = F
  mapping = mapping
  opt_score = ["_maxmu", "_X"]
  # Load structure ----
  NET = WAVES[nature](
    linkage, mode,
    nlog10 = nlog10,
    lookup = lookup,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut
  )
  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.C, nlog10, lookup, prob
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  ## Hierarchy object!! ----
  H = Hierarchy(
    NET, NET.C, R, NET.D,
    NET.nodes, linkage, mode, lookup=lookup
  )
  ## Compute features ----
  H.BH_features_parallel()
  ## Compute link entropy ----
  H.link_entropy_cpp("short", cut=cut)
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  ## Compute node entropy ----
  H.node_entropy_cpp("short", cut=cut)
  ## Update entropy ----
  H.entropy = [
    H.node_entropy, H.node_entropy_H,
    H.link_entropy, H.link_entropy_H
  ]
  # Set labels to network ----
  L = colECoG(NET)
  H.set_colregion(L)
  for score in opt_score:
    print(f"Find node partition using {score}")
    # Get best K and R ----
    K, R = get_best_kr(score, H)
    r = R[K == np.min(K)][0]
    k = K[K == np.min(K)][0]
    H.set_kr(k, r, score=score)
    print("\n\tBest K: {}\nBest R: {}\n".format(k, r))
    rlabels = get_labels_from_Z(H.Z, r)
    # Overlap ----
    NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
    cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
    H.set_cover(cover, score)
  save_class(
    H, NET.pickle_path,
    "hanalysis"
  )
  print("End!")
    # #@@ Todo: