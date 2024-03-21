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
from networks.MAC.shuffle import SHUFFLE
from various.data_transformations import maps
from modules.discovery import discovery_channel
from various.network_tools import *

def worker_shuffle(
  number_of_iterations : int, number_of_inj : int,
  number_of_nodes : int, total_number_nodes : int, data_version,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool,
  topology : str, mapping : str, index : str, discovery : str, mode : str
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  linkage = "single"
  structure = "FLN"
  nature = "original"
  distance = "tracto16"
  mode = mode
  alpha = 0.
  imputation_method = ""
  opt_score = ["_S"]  
  # Declare global variables DISTBASE ----
  __inj__ = number_of_inj
  __nodes__ = number_of_nodes
  __version__ = data_version
  # Print summary ----
  print(f"Number of iterations: {MAXI}")
  print("For NET parameters:")
  print(
    "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup : {}\ncut: {}".format(
      linkage, mode, opt_score, nlog10, lookup, cut
    )
  )
  print(f"topology: {topology}")
  print("Load MAC data ----")
  # Create macaque class ----
  NET = MAC[f"MAC{__inj__}"](
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
    cut=cut, b=0., alpha=alpha,
    discovery = discovery
  )
  # Load hierarhical analysis ----
  NET_H = read_class(
    NET.pickle_path,
    "hanalysis"
  )
  # Create colregions ----
  L = colregion(NET, labels_name=f"labels{__inj__}")
  labels_default = L.labels.copy()

  # Create hrh class ----
  data = HRH(NET_H, L, MAXI)
  RAND_H = 0

  # RANDOM networks ----
  serie = np.arange(MAXI)
  print("Create random networks ----")
  for i in serie:
    print("Iteration: {}".format(i))
    # Add iteartion to data ----
    data.set_iter(i)
    # Create network ----
    print("Shuffle the order of the nodes")
    RAND = SHUFFLE(
      __inj__, total_number_nodes,
      linkage, mode, i,
      structure = structure,
      version = __version__,
      nlog10=nlog10, lookup=lookup, cut=cut,
      topology=topology, distance=distance,
      mapping=mapping, index=index,
      discovery=discovery
    )
    RA = NET.A.copy()
    RD = NET.D.copy()
    ## Randomized Network ----
    perm_ec = np.random.permutation(NET.nodes)
    perm_non_ec = np.random.permutation(np.arange(NET.nodes, NET.rows))
    perm = list(perm_ec) + list(perm_non_ec)
    RA = RA[:, perm_ec][perm, :]
    RD = RD[:, perm_ec][perm, :]
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RA, nlog10, lookup, prob, b=0.
    )
    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    RAND_H = Hierarchy(
      RAND, RA[:, :__nodes__], R[:, :__nodes__], RD,
      __nodes__, linkage, mode, lookup=lookup, alpha=alpha,
      index=index
    )
    ## Compute features ----
    RAND_H.BH_features_cpp_no_mu()
    ## Compute lq arbre de merde ----
    RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
    # Set colregion ----
    L.labels = labels_default[perm]
    RAND_H.set_colregion(L)
    # Stats ----
    data.set_data_measurements_zero(RAND_H, i)
    data.set_hierarchical_association(RAND_H.Z, i, perm=(True, perm_ec))
    data.set_stats(RAND_H)
    for SCORE in opt_score:
      # Get k from RAND_H ----
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
          data.set_clustering_similarity(rlabels2[invert_permutation(perm_ec)], cover, SCORE, direction)
          data.set_overlap_data_zero(ocn, SCORE, direction)
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H, bias=0.)
    data.set_pickle_path(RAND_H, bias=0.)
    print("Save data")
    print(data.pickle_path)
    save_class(
      data, data.pickle_path,
      f"series_{MAXI}"
    )
  print("End")