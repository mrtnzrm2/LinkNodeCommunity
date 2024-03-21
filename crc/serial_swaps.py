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
from networks.structure import STR
from networks.swapnet import SWAPNET
from various.network_tools import *
from modules.discovery import discovery_channel
from various.data_transformations import maps

def worker_swaps(
  subject : str, number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, total_number_of_nodes : int , data_version, distance_matrix_name :str,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, discovery : str, bias : float, mode : str
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
  mode = mode
  alpha = 0.
  structure = "FLNe"
  distance = distance_matrix_name
  nature = "original"
  imputation_method = ""
  opt_score = ["_S"]
  sln = T
  # Declare global variables DISTBASE ----
  __inj__ = number_of_inj
  __nodes__ = number_of_nodes
  __version__ = data_version
  __model__ = "TWOMIX_FULL"
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
    "nodes: {}\ninj: {}".format(str(__nodes__),str(__inj__))
  )
  # Start main ----
  print("Load data ----")
  # Create macaque class ----
  NET = STR[f"{subject}{__inj__}"](
    linkage, mode,
    structure = structure,
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
    b=bias, alpha=alpha,
    discovery = discovery
  )
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
  # RANDOM networks ----
  serie = np.arange(MAXI)
  print("Create random networks ----")
  for i in serie:
    print("***\tIteration: {}".format(i))
    data.set_iter(i)
    RAND = SWAPNET(
      __inj__,
      total_number_of_nodes,
      linkage,
      mode, i,
      structure = structure,
      version = __version__,
      topology=topology,
      nature = nature,
      distance = distance,
      model = __model__,
      mapping=mapping,
      index=index,
      nlog10 = nlog10, lookup = lookup,
      cut=cut, b=bias, discovery=discovery,
      subject=subject
    )
    RAND.C, RAND.A = NET.C, NET.A
    RAND.D = NET.D

    # Create network ----
    print("Create random graph")
    RAND.random_one_k_TWOMX(NET.SLN, run=run, swaps=100000, on_save_csv=F)   #****
    # RAND.random_one_k_dense(run=run, swaps=100000, on_save_csv=F)   #****
    # RAND.random_dir_weights(run=run, swaps=100000, on_save_csv=F)   #****
    # RAND.random_one_k(run=run, swaps=100000, on_save_csv=F)

    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RAND.A, nlog10, lookup, prob, b=bias
    )
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, RAND.A[:, :__nodes__], R[:, :__nodes__], RAND.D,
      __nodes__, linkage, mode, lookup=lookup, alpha=alpha,
      index=index
    )
    ## Compute features ----
    RAND_H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    RAND_H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    RAND_H.get_h21merge()
    ## Compute node entropy ----
    RAND_H.node_entropy_cpp("short", cut=cut)
    # Set colregion ----
    RAND_H.set_colregion(L)
    RAND_H.delete_dist_matrix()
    # Stats ----
    data.set_data_measurements_zero(RAND_H, i)
    data.set_hierarchical_association(RAND_H.Z, i)
    data.set_stats(RAND_H)
    # Set entropy ----
    data.set_entropy_zero(
      [RAND_H.node_entropy, RAND_H.node_entropy_H,
       RAND_H.link_entropy, RAND_H.link_entropy_H],  
    )
    data.set_stats(RAND_H)
    for SCORE in opt_score:
      # Get best k, r for given score ----
      K, R, _ = get_best_kr_equivalence(SCORE, RAND_H)
      for k, r in zip(K, R):
        RAND_H.set_kr(k, r, SCORE)
        data.set_kr_zero(RAND_H)
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        rlabels = skim_partition(rlabels)
        # Overlap ----
        for direction in ["source", "target", "both"]:
          print("***", direction)
          ocn, subcover, _, rlabels2 = discovery_channel[discovery](
            RAND_H, k, rlabels, direction=direction, index=index
          )
          cover = omega_index_format(rlabels2, subcover, RAND_H.colregion.labels[:RAND_H.nodes])
          data.set_association_zero(SCORE, cover, direction)
          data.set_clustering_similarity(rlabels2, cover, SCORE, direction)
          data.set_overlap_data_zero(ocn, SCORE, direction)
          if direction == "both" and sln:
            R_data = len(NET_H.cover[direction][SCORE])
            random_partition = random_partition_R(NET.nodes, R_data)

            data_sln = RAND_H.get_data_firstmerge(
              RAND.B, NET_H.cover[direction][SCORE], RAND_H.colregion.labels[:RAND_H.nodes]
            )

            data.set_sln_matrix_zero(
              RAND_H.get_sln_matrix(data_sln, NET_H.cover[direction][SCORE]), key="conf"
            )

            data_sln = RAND_H.get_data_firstmerge(
              NET.SLN, random_partition, RAND_H.colregion.labels[:RAND_H.nodes]
            )

            data.set_sln_matrix_zero(RAND_H.get_sln_matrix(data_sln, random_partition), key="shuffle")
  # Save ----
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
  