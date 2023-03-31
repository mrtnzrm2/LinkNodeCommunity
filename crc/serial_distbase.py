# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libraries ----
import numpy as np
# Import network libraries ----
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from networks_serial.hrh import HRH
from networks.structure import MAC
from networks.distbase import DISTBASE
from various.data_transformations import maps
from various.network_tools import *
from various.fit_tools import fitters

def worker_distbase(
  number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, total_number_nodes : int, data_version, distbase : str,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, bias : float
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
  structure = "FLN"
  nature = "original"
  distance = "MAP3D"
  mode = "ALPHA"
  imputation_method = ""
  opt_score = ["_maxmu", "_X", "_D"]  
  # Statistic test ----
  alternative = "less"
  # Declare global variables DISTBASE ----
  __inj__ = number_of_inj
  __nodes__ = number_of_nodes
  __version__ = data_version
  __model__ = distbase
  __bin__ = 12
  # Print summary ----
  print(f"Number of iterations: {MAXI}")
  print("For NET parameters:")
  print(
    "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup : {}\ncut: {}".format(
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
    "distbase: {}\nnodes: {}\ninj: {}\nalternative: {}".format(
      __model__, str(__nodes__),str(__inj__), alternative
    )
  )
  print("Load MAC data ----")
  # Create macaque class ----
  NET = MAC(
    linkage, mode,
    structure = structure,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    nature=nature,
    model=imputation_method,
    distance=distance,
    inj=__inj__,
    topology=topology,
    index=index, mapping=mapping,
    cut=cut, b=bias
  )
  _, _, _, _, est = fitters[__model__](NET.D, NET.C, NET.nodes, __bin__)
  lb = est.coef_[0]
  # Load hierarhical analysis ----
  NET_H = read_class(
    NET.pickle_path,
    "hanalysis"
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
    print("Iteration: {}".format(i))
    # Add iteartion to data ----
    data.set_iter(i)
    RAND = DISTBASE(
        __inj__, total_number_nodes,
        linkage, __bin__, mode, i,
        structure = structure,
        version = __version__, model=distbase,
        nlog10=nlog10, lookup=lookup, cut=cut,
        topology=topology, distance=distance,
        mapping=mapping, index=index, b=bias,
        lb=lb
      )
    # Create distance matrix ----
    D = RAND.get_distance_matrix(NET.struct_labels)
    # Create network ----
    print("Create random graph")
    RC = RAND.distbase_dict[__model__](
      D, NET.C, run=run, on_save_csv=F
    )
    G = column_normalize(RC)
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      G, nlog10, lookup, prob, b=bias
    )
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, G[:, :__nodes__], R[:, :__nodes__], D,
      __nodes__, linkage, mode, lookup=lookup
    )
    ## Compute features ----
    RAND_H.BH_features_parallel()
    ## Compute link entropy ----
    RAND_H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    RAND_H.set_colregion(L)
    # Stats ----
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
      [node_entropy, node_entropy_H, RAND_H.link_entropy, RAND_H.link_entropy_H],  
    )
    for score in opt_score:
      # Get k from RAND_H ----
      K, R = get_best_kr(score, RAND_H)
      r = R[K == np.min(K)][0]
      k = K[K == np.min(K)][0]
      RAND_H.set_kr(k, r, score)
      data.set_kr_zero(RAND_H)
      rlabels = get_labels_from_Z(RAND_H.Z, r)
      # Overlap ----
      ocn, subcover = RAND_H.get_ocn_discovery(k, rlabels)
      cover = omega_index_format(
        rlabels, subcover, RAND_H.colregion.labels[:RAND_H.nodes]
      )
      data.set_clustering_similarity(rlabels, cover, score)
      data.set_overlap_data_zero(ocn, score)
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H, bias=bias)
    data.set_pickle_path(RAND_H, bias=bias)
    print("Save data")
    save_class(
      data, data.pickle_path,
      f"series_{MAXI}"
    )
  print("End")