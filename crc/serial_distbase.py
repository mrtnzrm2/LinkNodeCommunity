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
from networks_serial.hrh import HRH
from networks.structure import STR
from networks.distbase import DISTBASE
from various.data_transformations import maps
from modules.discovery import discovery_channel
from various.network_tools import *
from various.fit_tools import fitters

def worker_distbase(
  subject : str, number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, total_number_nodes : int, data_version, distance_matrix : str, distbase : str,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, discovery : str, bias : float, bins : int, mode : str
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
  structure = "FLNe"
  nature = "original"
  fitter = "EXPTRUNC"
  distance = distance_matrix
  mode = mode
  alpha = 0.
  imputation_method = ""
  opt_score = ["_S"]  
  # Statistic test ----
  alternative = "less"
  # Declare global variables DISTBASE ----
  __inj__ = number_of_inj
  __nodes__ = number_of_nodes
  __version__ = data_version
  __model__ = distbase
  __bin__ = bins
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
  print("Load data ----")
  # Create macaque class ----
  NET = STR[f"{subject}{__inj__}"](
    linkage, mode,
    structure=structure,
    nlog10=nlog10, lookup=lookup,
    version=__version__,
    nature=nature,
    model=imputation_method,
    distance=distance,
    inj=__inj__,
    topology=topology,
    index=index, mapping=mapping,
    cut=cut, b=bias, alpha=alpha,
    discovery=discovery
  )
  pars, _, _, _, _ = fitters[fitter](NET.D, NET.CC, __bin__)
  lb = 1 / pars[2]
  loc = np.min(NET.D[NET.D > 0])
  if __model__ == "M":
    loc = 0
  b = np.max(NET.D)

  # Load hierarhical analysis ----
  NET_H = read_class(
    NET.pickle_path,
    "hanalysis"
  )
  # Create colregions ----
  L = colregion(NET, labels_name=f"labels{__inj__}")
  # Create hrh class ----
  data = HRH(NET_H, L, MAXI)
  RAND_H = 0
  # Get distance matrix from structure ----
  D = NET.D
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
      subject=subject,
      version = __version__, model=distbase, fitter=fitter,
      nlog10=nlog10, lookup=lookup, cut=cut,
      topology=topology, distance=distance,
      mapping=mapping, index=index, b=bias,
      lb=lb, discovery=discovery,
      rho=adj2Den(NET.A[:NET.nodes,:][:, :NET.nodes])
    )

    RAND.rows = NET.rows
    # Create network ----
    print("Create random graph")
    RC = RAND.distbase_dict[__model__](
      D, NET.CC, loc=loc, b=b, run=run, on_save_csv=F
    )
    RA = column_normalize(RC)
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RA, nlog10, lookup, prob, b=bias
    )
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, RA[:, :__nodes__], R[:, :__nodes__], D,
      __nodes__, linkage, mode, lookup=lookup, alpha=alpha,
      index=index
    )
    ## Compute features ----
    RAND_H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    RAND_H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    ## Compute node entropy ----
    RAND_H.node_entropy_cpp("short", cut=cut)
    # Set colregion ----
    RAND_H.set_colregion(L)
    # Stats ----
    data.set_data_measurements_zero(RAND_H, i)
    data.set_hierarchical_association(RAND_H.Z, i)
    # data.set_hp(NET_H.hp, NET_H.ehmi, RAND_H.Z, RAND_H.nodes)
    data.set_stats(RAND_H)
    # Set entropy ----
    data.set_entropy_zero(
      [RAND_H.node_entropy, RAND_H.node_entropy_H,
       RAND_H.link_entropy, RAND_H.link_entropy_H],  
    )
    for SCORE in opt_score:
      # Get k from RAND_H ----
      K, R, _ = get_best_kr_equivalence(SCORE, RAND_H)
      for k, r in zip(K, R):
        RAND_H.set_kr(k, r, SCORE)
        data.set_kr_zero(RAND_H)
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        rlabels = skim_partition(rlabels)
        # Overlap ----
        for direction in ["both"]: # "source", "target", 
          print("***", direction)
          noc, subcover, _, rlabels2 = discovery_channel[discovery](
            RAND_H, k, rlabels, direction=direction, index=index
          )
          cover = omega_index_format(rlabels2, subcover, RAND_H.colregion.labels[:RAND_H.nodes])
          data.set_association_zero(SCORE, cover, direction)
          data.set_clustering_similarity(rlabels2, cover, SCORE, direction)
          data.set_overlap_data_zero(noc, SCORE, direction)
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H)
    data.set_pickle_path(RAND_H)
    print("Save data")
    print(data.pickle_path)
    save_class(
      data, data.pickle_path,
      f"series_{MAXI}"
    )
  print("End")