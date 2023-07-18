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
topologies = ["MIX"]
bias = [0]
indices = ["D1_2_4"]
modes = ["ZERO"]
discovery = ["discovery_5"]
list_of_lists = itertools.product(
  *[cut, modes, topologies, indices, bias, discovery]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
structure = "LN"
distance = "tracto16"
nature = "original"
mapping = "trivial"
alpha = 0.
imputation_method = ""
opt_score = ["_X", "_S", "_SD"]
version = "57d106"
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  for _cut_, mode, topology, index, bias, disco in list_of_lists:
    bias = float(bias)
    if _cut_ == "True":
      cut = T
    else: cut = F
    # Load structure ----
    NET = MAC[f"MAC{__inj__}"](
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
      cut=cut, b = bias, alpha=alpha,
      discovery = disco
    )
    NET.create_pickle_directory()
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      NET.C, nlog10, lookup, prob, b=bias
    )
    # Compute Hierarchy ----
    print("Compute Hierarchy")
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.C, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup, alpha=alpha
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    H.entropy = [
      H.node_entropy, H.node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__inj__}")
    H.set_colregion(L)
    H.delete_dist_matrix()
    # Entropy ----
    HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
    HS.Z2dict("short")
    HS.zdict2newick(HS.tree, weighted=F, on=T)
    HS.zdict2newick(HS.tree, weighted=T, on=T)
    ial = 0
    for SCORE in opt_score:
      # Get best K and R ----
      K, R = get_best_kr_equivalence(SCORE, H)
      for k, r in zip(K, R):
        print(f"Find node partition using {SCORE}")
        print("\n\tBest K: {}\nBest R: {}\n".format(k, r))
        rlabels = get_labels_from_Z(H.Z, r)
        # Overlap ----
        NET.overlap, NET.data_nocs = H.discovery_channel[disco](k, rlabels)
        print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
        cover = omega_index_format(rlabels,  NET.data_nocs, NET.struct_labels[:NET.nodes])
        # Set communitry structure ----
        H.set_kr(k, r, score=SCORE)
        H.set_overlap_labels(NET.overlap, SCORE)
        H.set_cover(cover, SCORE)
    save_class(
      H, NET.pickle_path,
      "hanalysis"
    )
  print("End!")