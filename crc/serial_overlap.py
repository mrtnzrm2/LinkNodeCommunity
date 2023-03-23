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
from networks_serial.overlaphrh import OVERLAPHRH
from networks.overlapping import OVERLAPPING
from modules.colregion import colregion
from various.network_tools import *

def worker_overlap(
  number_of_iterations : int, number_of_nodes : int,
  nlog10 : bool, lookup : bool, prob : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str,
  kav : float, maxk : int, mut : float, muw : float, beta : float,
  t1 : float, t2 : float,  nmin : int, nmax : int, on :int, om : int
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  __nodes__ = number_of_nodes
  linkage = "single"
  __mode__ = "ALPHA"
  opt_score = ["_maxmu", "_X", "_D"]
  # Overlapping WDN paramters ----
  opar = {
    "-N" : "{}".format(
      str(__nodes__)
    ),
    "-k" : f"{kav}",
    "-maxk" : f"{maxk}",
    "-mut" : f"{mut}",
    "-muw" : f"{muw}",
    "-beta" : f"{beta}",
    "-t1" : f"{t1}",
    "-t2" : f"{t2}",
    "-nmin" : f"{nmin}",
    "-nmax" : f"{nmax}",
    "-on" : f"{on}",
    "-om" : f"{om}"
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
      nlog10=nlog10, lookup=lookup,
      topology=topology,
      cut=cut,
      mapping=mapping,
      index=index,
      parameters = opar
    )
    # Create network ----
    print("Create random graph")
    RAND.random_WDN_overlap_cpp(
      run=run, on_save_pickle = F   #****
    )
    if np.sum(np.isnan(RAND.A)) > 0:
      print(
        "LFB failed to create the network with the desired properties."
      )
    else:   
      RAND.col_normalized_adj(on=F)
      L = colregion(RAND)
      RAND.set_colregion(L)
      # Compute RAND Hierarchy ----
      print("Compute Hierarchy")
      RAND_H = Hierarchy(
        RAND, RAND.A, RAND.A, zeros(RAND.A.shape),
        __nodes__, linkage, __mode__
      )
      ## Compute features ----
      RAND_H.BH_features_parallel()
      ## Compute lq arbre de merde ----
      RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
      RAND_H.set_colregion(L)
      # Save stats ----
      data.set_data_measurements(RAND_H, i)
      for score in opt_score:
        # Get best k, r for given score ----
        K, R = get_best_kr(score, RAND_H)
        for ii, kr in enumerate(zip(K, R)):
          k, r = kr
          # Single linkage part ----
          print("Best K: {}\nBest R: {}".format(k, r))
          rlabels = get_labels_from_Z(RAND_H.Z, r)
          nocs, noc_covers = RAND_H.get_ocn_discovery(k, rlabels)
          sen, sep = RAND.overlap_score_discovery(
            k, nocs, RAND_H.colregion.labels[:RAND_H.nodes], on=T
          )
          omega = RAND.omega_index(
            rlabels, noc_covers, RAND_H.colregion.labels[:RAND_H.nodes], on=T
          )
          # NMI between ground-truth and pred labels ----
          data.set_nmi_nc_overlap(
            RAND.labels, rlabels, RAND.overlap, noc_covers, omega,
            score = score
          )
          data.set_overlap_scores(
            omega, sen, sep, score = score
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