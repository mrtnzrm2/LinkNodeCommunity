# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Import libraries ----
from modules.hierarmerge import Hierarchy
from modules.hierarentropy import Hierarchical_Entropy
from networks_serial.scalehrh import SCALEHRH
from networks.scalefree import SCALEFREE
from various.network_tools import *
from modules.colregion import colregion
from numpy import zeros

def worker_scalefree(
  number_of_iterations : int, number_of_nodes : int, benchmark : str, mode : str,
  nlog10 : bool, lookup : bool, cut : bool, run : bool,
  topology : str, mapping : str, index : str,
  kav : float, maxk : int, mut : float, muw :float, beta : float,
  t1 : float, t2 : float, nmin : int, nmax : int, **kwargs
):
  # Declare global variables NET ----
  MAXI = number_of_iterations
  __nodes__ = number_of_nodes
  linkage = "single"
  discovery = "discovery_7"
  __mode__ =  mode
  alpha = 0.
  opt_score = ["_X", "_D", "_S"]
  # WDN paramters ----
  par = {
    "-N" : f"{__nodes__}",
    "-k" : f"{kav}",
    "-maxk" : f"{maxk}",
    "-mut" : f"{mut}",
    "-muw" : f"{muw}",
    "-beta" : f"{beta}",
    "-t1" : f"{t1}",
    "-t2" : f"{t2}",
    "-nmin" : f"{nmin}",
    "-nmax" : f"{nmax}",
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
  print(par)
  # Start main ----
  # Store data ---
  data = SCALEHRH(linkage)
  RAND_H = 0
  # RANDOM networks ----
  print("Create random networks ----")
  i = 0
  while i < MAXI:
    print("***\tIteration: {}".format(i))
    # Add iteartion to data ----
    data.set_iter(i)
    RAND = SCALEFREE(
      i,
      linkage,
      __mode__,
      benchmark=benchmark,
      nlog10=nlog10, lookup=lookup,
      topology=topology,
      cut=cut,
      mapping=mapping,
      index=index,
      parameters = par,
      discovery = discovery,
      **kwargs
    )
    # Create network ----
    print("Create random graph")
    if benchmark == "WDN":
      RAND.random_WDN_cpp(run=run, random_seed=T, on_save_pickle=F)     #*****
    elif benchmark == "WN":
      RAND.random_WN_cpp(run=run, random_seed=T, on_save_pickle=F)
    RAND.col_normalized_adj(on=F)
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
    # Save stats ----
    data.set_data_measurements(RAND_H, i)
    # Set labels to network ----
    L = colregion(RAND)
    RAND_H.set_colregion(L)
    RAND_H.delete_dist_matrix()
    #  Update entropy ----
    # data.update_entropy(
    #   [RAND_H.node_entropy, RAND_H.node_entropy_H,
    #    RAND_H.link_entropy, RAND_H.link_entropy_H],  
    # )
    for score in opt_score:
      print("Score: {}".format(score))
      # Get best k, r for given score ----
      K, R, _ = get_best_kr_equivalence(score, RAND_H)
      for k, r in zip(K, R):
        print("Score: {}".format(score))
        # Single linkage part ----
        print("Best K: {}\nBest R: {}".format(k, r))
        rlabels = get_labels_from_Z(RAND_H.Z, r)        
        data.set_nmi_nc(RAND.labels, rlabels, score=score)

    if np.sum(np.isnan(rlabels)) == 0: i += 1
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