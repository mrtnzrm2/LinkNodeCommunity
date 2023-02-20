# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libraries ----
from numpy import arange
# Import network libraries ----
from modules.hierarmerge import Hierarchy
from networks_serial.hrh import HRH
from various.network_tools import *
from modules.colregion import colregion
from networks.structure import MAC
from networks.distbase import DISTBASE

def worker_distbase(
  number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, total_number_nodes : int, data_version, distbase : str,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, bias : float
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
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
  for score in opt_score:
    overlap_labels = NET_H.overlap.labels.loc[
      NET_H.overlap.score == score
    ].to_numpy()
    data.set_overlap_data_one(overlap_labels, score)
    data.set_nodes_labels_single(NET_H, score)
  # RANDOM networks ----
  serie = arange(MAXI)
  print("Create random networks ----")
  for i in serie:
    print("Iteration: {}".format(i))
    # Add iteartion to data ----
    data.set_iter(i)
    RAND = DISTBASE(
        __inj__, total_number_nodes,
        linkage, __bin__, mode, i,
        version = __version__, model=distbase,
        nlog10=nlog10, lookup=lookup, cut=cut,
        topology=topology, distance=distance,
        mapping=mapping, index=index, b=bias,
        lb=0.07921125
      )
    # Create distance matrix ----
    D = RAND.get_distance_matrix(NET.struct_labels)
    # Create network ----
    print("Create random graph")
    RC = RAND.distbase_dict[__model__](
      D, NET.C, run=run, on_save_csv=F
    )
    R = column_normalize(RC)
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, R[:, :__nodes__], D,
      __nodes__, linkage, mode,
      prob=prob, b=bias
    )
    ## Compute features ----
    RAND_H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    RAND_H.set_colregion(L)
    # Stats ----
    data.set_data_homogeneity_zero(RAND_H.R)
    data.set_data_measurements_zero(RAND_H, i)
    save_class(
      RAND_H, RAND.pickle_path,
      "hanalysis_{}".format(RAND_H.subfolder),
      on=F
    )
    # Convergence ----
    data.set_stats(RAND_H)
    for score in opt_score:
      # Get k from RAND_H ----
      k, r = get_best_kr_equivalence(score, RAND_H)
      RAND_H.set_kr(k, r, score)
      data.set_kr_zero(RAND_H)
      rlabels = get_labels_from_Z(RAND_H.Z, r)
      data.set_nmi_nc(rlabels[:__inj__], score)
      # Overlap ----
      ocn, _ = RAND_H.get_ocn_discovery(k, rlabels)
      data.set_overlap_data_zero(ocn, score)
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H)
    data.set_pickle_path(RAND_H)
    print("Save data")
    save_class(
      data, data.pickle_path,
      "series_{}".format(
        number_of_iterations
      )
    )
  print("End")