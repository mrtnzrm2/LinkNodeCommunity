# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import numpy as np
# Personal libs ----
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from networks_serial.hrh import HRH
from networks.structure import MAC
from networks.swapnet import SWAPNET
from various.network_tools import *
from various.data_transformations import maps

def worker_swaps(
  number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, data_version,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, bias : float
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
  mode = "ALPHA"
  distance = "MAP3D"
  nature = "original"
  imputation_method = ""
  opt_score = ["_maxmu", "_X"]
  # Declare global variables DISTBASE ----
  __inj__ = number_of_inj
  __nodes__ = number_of_nodes
  __version__ = data_version
  __model__ = "1k"
  # T test ----
  alternative = "less"
  # Print summary ----
  print(f"Number of iterations: {MAXI}")
  print("For NET parameters:")
  print(
    "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup: {}\ncut: {}".format(
      linkage, mode, opt_score, nlog10, lookup, cut
    )
  )
  print(f"topology: {topology}")
  print("For imputation parameters:")
  print(
    "nature: {}\nmodel: {}".format(
      nature, imputation_method
    )
  )
  print("Random network and statistical paramteres:")
  print(
    "nodes: {}\ninj: {}\nalternative: {}".format(
      str(__nodes__),str(__inj__), alternative
    )
  )
  # Start main ----
  print("Load MAC data ----")
  # Create macaque class ----
  NET = MAC(
    linkage, mode,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    distance = distance,
    nature = nature,
    model = imputation_method,
    inj= __inj__,
    topology= topology,
    mapping=mapping,
    index=index,
    cut = cut,
    b=bias
  )
  # Load hierarhical analysis ----
  NET_H = read_class(
    NET.pickle_path,
    "hanalysis_{}".format(NET.subfolder),
  )
  # Create colregions ----
  L = colregion(NET)
  # Create hrh class ----
  data = HRH(NET_H, L)
  RAND_H = 0
  # RANDOM networks ----
  serie = np.arange(MAXI)
  print("Create random networks ----")
  for i in serie:
    print("***\tIteration: {}".format(i))
    data.set_iter(i)
    RAND = SWAPNET(
      __inj__,
      linkage,
      mode, i,
      version = __version__,
      topology=topology,
      nature = nature,
      distance = distance,
      model = __model__,
      mapping=mapping,
      index=index,
      nlog10 = nlog10, lookup = lookup,
      cut=cut, b=bias
    )
    # Create network ----
    print("Create random graph")
    RAND.random_one_k(run=run, on_save_csv=F)   #****
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RAND.A, nlog10, lookup, prob, b=bias
    )
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, RAND.A[:, :__nodes__], R[:, :__nodes__], RAND.D,
      __nodes__, linkage, mode, lookup=lookup
    )
    ## Compute features ----
    RAND_H.BH_features_cpp()
    ## Compute link entropy ----
    RAND_H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    RAND_H.set_colregion(L)
    # Saving statistics ----
    data.set_data_homogeneity_zero(RAND_H.R)
    data.set_data_measurements_zero(RAND_H, i)
    data.set_stats(RAND_H)
    # Entropy ----
    HS = Hierarchical_Entropy(
      RAND_H.Z, RAND_H.nodes, RAND_H.colregion.labels[:RAND_H.nodes]
    )
    HS.Z2dict("short")
    HS.zdict2newick(HS.tree, weighted=F, on=F)
    HS.zdict2newick(HS.tree, weighted=T, on=F)
    node_entropy = HS.S(HS.tree)
    node_entropy_H = HS.S_height(HS.tree)
    data.set_entropy_zero(
      [node_entropy, node_entropy_H, RAND_H.link_entropy, RAND_H.link_entropy_H]
    )
    for score in opt_score:
      # Get best k, r for given score ----
      k, r = get_best_kr(score, RAND_H)
      RAND_H.set_kr(k, r, score)
      data.set_kr_zero(RAND_H)
      # Add iteartion to data----
      rlabels = get_labels_from_Z(RAND_H.Z, r)
      # Overlap ----
      ocn, subcover = RAND_H.get_ocn_discovery(k, rlabels)
      cover = omega_index_format(
        rlabels, subcover, RAND_H.colregion.labels[:RAND_H.nodes]
      )
      data.set_clustering_similarity(rlabels, cover, score)
      data.set_overlap_data_zero(ocn, score)
  # Save ----
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H, bias=bias)
    data.set_pickle_path(RAND_H, bias=bias)
    print("Save data")
    print(data.pickle_path)
    save_class(
      data, data.pickle_path,
      "series_{}".format(number_of_iterations)
    )
  print("End")
  