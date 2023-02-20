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
from various.network_tools import *
from networks.structure import MAC
# Iterable varaibles ----
cut = [F]
topologies = ["MIX", "TARGET", "SOURCE"]
list_of_lists = itertools.product(
  *[cut, topologies]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
distance = "MAP3D"
nature = "original"
mapping = "R2"
index = "jacw"
mode = "ALPHA"
imputation_method = ""
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
version = 220830
__nodes__ = 57
__inj__ = 57

# Start main ----
if __name__ == "__main__":
  for _cut_, topology in list_of_lists:
    if _cut_ == "True":
      cut = T
    else: cut = F
    # Load structure ----
    NET = MAC(
      linkage, mode,
      nlog10=nlog10, lookup=lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      topology=topology,
      index=index, mapping=mapping,
      cut=cut
    )
    # Compute Hierarchy ----
    print("Compute Hierarchy")
    # Save ----
    if save_data:
      ## Hierarchy object!! ----
      H = Hierarchy(
        NET, NET.A, NET.D,
        __nodes__, linkage, mode,
        prob=prob
      )
      ## Compute features ----
      H.BH_features_cpp()
      ## Compute lq arbre de merde ----
      H.la_abre_a_merde_cpp(H.BH[0])
      # Set labels to network ----
      L = colregion(NET)
      H.set_colregion(L)
      # Save ----
      save_class(
        H, NET.pickle_path,
        "hanalysis_{}".format(H.subfolder),
        on=F
      )
    else:
      H = read_class(
        NET.pickle_path,
        "hanalysis_{}".format(NET.subfolder),
        on=F
      )
    for score in opt_score:
      print(f"Find node partition using {score}")
      # Get best K and R ----
      k, r = get_best_kr_equivalence(score, H)
      H.set_kr(k, r, score=score)
      rlabels = get_labels_from_Z(H.Z, r)
      # Overlap ----
      NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
      H.set_overlap_labels(NET.overlap, score)
      print(
        "Areas with predicted overlapping communities:\n",
        NET.overlap
      )
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
    )
  print("End!")