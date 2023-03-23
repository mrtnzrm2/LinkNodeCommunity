# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import itertools
# Import libraries ---- 
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from networks.structure import MAC
from various.data_transformations import maps
from various.network_tools import *
# Iterable varaibles ----
cut = [F]
topologies = ["MIX", "TARGET", "SOURCE"]
bias = [1e-5, 1e-2, 0.1, 0.3, 0.5]
list_of_lists = itertools.product(
  *[cut, topologies, bias]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
structure = "FLN"
distance = "MAP3D"
nature = "original"
mapping = "R2"
index = "jacw"
mode = "ALPHA"
imputation_method = ""
opt_score = ["_maxmu", "_X"]
version = 220830
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  for _cut_, topology, bias in list_of_lists:
    bias = float(bias)
    if _cut_ == "True":
      cut = T
    else: cut = F
    # Load structure ----
    NET = MAC(
      linkage, mode,
      structure = structure,
      nlog10=nlog10, lookup=lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      topology=topology,
      index=index, mapping=mapping,
      cut=cut, b = bias
    )
    NET.create_pickle_directory()
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      NET.A, nlog10, lookup, prob, b=bias
    )
    # Compute Hierarchy ----
    print("Compute Hierarchy")
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET)
    H.set_colregion(L)
    # Entropy ----
    HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
    HS.Z2dict("short")
    HS.zdict2newick(HS.tree, weighted=F, on=T)
    HS.zdict2newick(HS.tree, weighted=T, on=T)
    node_entropy = HS.S(HS.tree)
    node_entropy_H = HS.S_height(HS.tree)
    H.entropy = [
      node_entropy, node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    for score in opt_score:
      print(f"Find node partition using {score}")
      # Get best K and R ----
      k, r = get_best_kr(score, H)
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
      "hanalysis_{}".format(H.subfolder),
    )
  print("End!")