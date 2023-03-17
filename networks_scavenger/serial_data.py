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
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.hierarentropy import Hierarchical_Entropy
from networks.structure import MAC
from various.data_transformations import maps
from various.network_tools import *
# Iterable varaibles ----
cut = [F]
topologies = ["MIX", "TARGET", "SOURCE"]
bias = [0]
list_of_lists = itertools.product(
  *[cut, topologies, bias]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
distance = "MAP3D"
nature = "original"
mapping = "R4"
index = "simple"
mode = "ALPHA"
imputation_method = ""
opt_score = ["_maxmu", "_X"]
version = 220830
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  for _cut_, topology, bias in list_of_lists:
    bias = float(bias)
    if _cut_ == "True":
      cut = T
    else: cut = F
    # Load structure ----
    NET = MAC(
      linkage, mode,
      nlog10=nlog10, lookup=lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      topology=topology,
      index=index, mapping=mapping,
      cut=cut, b = bias
    )
    H = read_class(
    NET.pickle_path,
    "hanalysis_{}".format(NET.subfolder),
  )
    plot_h = Plot_H(NET, H)
    plot_h.plot_measurements_Entropy(on=T)
    plot_h.plot_measurements_D(on=T)
    plot_h.plot_measurements_mu(on=T)
    plot_h.plot_measurements_X(on=T)
    plot_n = Plot_N(NET, H)
    plot_n.plot_akis(NET.D, s=5, on=T)
    for score in opt_score:
      # Get best K and R ----
      k, r = get_best_kr(score, H)
      rlabels = get_labels_from_Z(H.Z, r)
      # Overlap ----
      NET.overlap, NET.data_nocs = H.get_ocn_discovery(k, rlabels)
      # Plot H ----
      plot_h.core_dendrogram([r], on=T) #
      plot_h.lcmap_pure([k], labels = rlabels, on=F) #
      plot_h.heatmap_pure(r, on=T, labels = rlabels) #
      plot_h.heatmap_dendro(r, on=F) #
      plot_h.lcmap_dendro([k], on=T) #
      plot_h.flatmap_dendro(
        NET, [k], [r], on=T, EC=T #
      )
  print("End!")