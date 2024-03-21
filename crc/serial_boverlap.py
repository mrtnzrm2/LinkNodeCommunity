# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Default libraties ----
from numpy import zeros
# Import libraries ----
from modules.hierarmerge import Hierarchy
from modules.hierarentropy import Hierarchical_Entropy
from networks_serial.overlaphrh import OVERLAPHRH
from networks.overlapping import OVERLAPPING
from modules.colregion import colregion
from various.network_tools import *
from modules.discovery import discovery_channel

def worker_boverlap(
  number_of_iterations : int, number_of_nodes : int, benchmark : str, mode : str,
  nlog10 : bool, lookup : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str, discovery : str,
  kav : float, maxk : int, mut : float, t1 : float, t2 : float,
  nmin : int, nmax : int, on :int, om : int, **kwargs
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  __nodes__ = number_of_nodes
  linkage = "single"
  __mode__ = mode
  alpha = 0.
  opt_score = ["_S", "_D"]
  # Overlapping WDN paramters ----
  opar = {
    "-N" : "{}".format(
      str(__nodes__)
    ),
    "-k" : f"{kav}",
    "-maxk" : f"{maxk}",
    "-mut" : f"{mut}",
    "-t1" : f"{t1}",
    "-t2" : f"{t2}",
    "-nmin" : f"{nmin}",
    "-nmax" : f"{nmax}",
    "-on" : f"{on}",
    "-om" : f"{om}",
    "-fixed_range" : T
  }
  # Print summary ----
  print(f"Number of iterations: {MAXI}")
  print("For NET parameters:")
  print(
    "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup: {}\ncut: {}".format(
      linkage, __mode__, opt_score, nlog10, lookup, cut
    )
  )
  print(f"topology: {topology}")
  print("WDN parameters:")
  print(opar)
  # Start main ----
  # Store data ---
  data = OVERLAPHRH(linkage)
  RAND_H = 0
  # RANDOM networks ----
  print("Create random networks ----")
  i = 0
  while i < MAXI:
    print("***\tIteration: {}".format(i))
    # Set iter to data ----
    data.set_iter(i)
    RAND = OVERLAPPING(
      i,
      linkage,
      __mode__,
      benchmark=benchmark,
      nlog10=nlog10, lookup=lookup,
      topology=topology,
      cut=cut,
      mapping=mapping,
      index=index,
      discovery=discovery,
      parameters = opar, **kwargs
    )
    # Create network ----
    print("Create random graph")
    if benchmark == "BN":
      RAND.random_BN_overlap_cpp(run=run, random_seed=T, on_save_pickle = F)
    if np.sum(np.isnan(RAND.A)) > 0:
      print("LFR failed to create the network with the desired properties.")
    else:   
      RAND.col_normalized_adj(on=F)
      L = colregion(RAND)
      RAND.set_colregion(L)
      # Compute RAND Hierarchy ----
      print("Compute Hierarchy")
      RAND_H = Hierarchy(
        RAND, RAND.A, RAND.A, zeros(RAND.A.shape),
        __nodes__, linkage, __mode__, alpha=alpha
      )
      ## Compute features ----
      RAND_H.BH_features_cpp_no_mu()
      ## Compute link entropy ----
      # RAND_H.link_entropy_cpp("short", cut=cut)
      ## Compute lq arbre de merde ----
      RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
      ## Compute node entropy ----
      # RAND_H.node_entropy_cpp("short", cut=cut)
      # Set colregion ----
      RAND_H.set_colregion(L)
      RAND_H.delete_dist_matrix()
      # Save stats ----
      data.set_data_measurements(RAND_H, i)
      # Update entropy ----
      # data.update_entropy(
      #   [RAND_H.node_entropy, RAND_H.node_entropy_H,
      #    RAND_H.link_entropy, RAND_H.link_entropy_H],  
      # )
      for score in opt_score:
        # Get best k, r for given score ----
        K, R, _ = get_best_kr_equivalence(score, RAND_H)
        for _, kr in enumerate(zip(K, R)):
          k, r = kr
          print("Score: {}".format(score))
          # Single linkage part ----
          print("Best K: {}\nBest R: {}".format(k, r))
          rlabels = get_labels_from_Z(RAND_H.Z, r)
          rlabels = skim_partition(rlabels)
          for direction in ["source"]:
            print(f"Direction {direction}")
            nocs, noc_covers, _, rlabels2 = discovery_channel[discovery](RAND_H, k, rlabels, direction=direction, index=index)
            sen, sep = RAND.overlap_score_discovery(
              k, nocs, RAND_H.colregion.labels[:RAND_H.nodes], on=T
            ) 
            omega = RAND.omega_index(
              rlabels2, noc_covers, RAND_H.colregion.labels[:RAND_H.nodes], on=T
            )
            # NMI between ground-truth and pred labels ----
            data.set_nmi_nc_overlap(
              RAND.labels, rlabels2, omega,
              score=score, direction=direction
            )
            data.set_overlap_scores(
              omega, sen, sep, score=score,
              direction=direction
            )
    i += 1
  # Save ----
  if isinstance(RAND_H, Hierarchy):
    data.set_subfolder(RAND_H.subfolder)
    data.set_plot_path(RAND_H)
    data.set_pickle_path(RAND_H)
    print("Save data")
    save_class(
      data, data.pickle_path,
      "series_{}".format(number_of_iterations)
    )
  print("End")