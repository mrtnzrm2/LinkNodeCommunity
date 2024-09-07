# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
## Standard libs ----
import numpy as np
import itertools
## Personal libs ----
from networks_serial.overlaphrh import OVERLAPHRH
from plotting_modules.plotting_o_serial_sf import PLOT_OS_SF
from various.network_tools import read_class
# Declare iter variables ----
topologies = ["TARGET", "SOURCE", "MIX"]
indices = ["Hellinger2"]
KAV = [7]
MUT = [0.1, 0.2, 0.3]
MUW = [0.01]
OM = [3]
list_of_lists = itertools.product(
  *[topologies, indices, KAV, MUT, MUW, OM]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables NET ----
MAXI = 10
__nodes__ = 100
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "ZERO"
opt_score = ["_S", "_S2", "_D"]
# Start main ----
if __name__ == "__main__":
  for topology, index, kav, mut, muw, om in list_of_lists:
    # Overlapping WDN paramters ----
    opar = {
      "-N" : "{}".format(
        str(__nodes__)
      ),
      "-k" : f"{kav}.0",
      "-maxk" : "50",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "5",
      "-nmax" : "25",
      "-on" : "5",
      "-om" : f"{om}"
    }
    # Print summary ----
    print("For NET parameters:")
    print(
      "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup: {}".format(
        linkage, __mode__, opt_score, nlog10, lookup
      )
    )
    print("WDN parameters:")
    print(opar)
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    data = read_class(
      "../pickle/RAN/scalefree/WDN/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/-fixed_range_True/{}/{}/{}/{}/{}".format(
        str(__nodes__),
        opar["-k"], opar["-maxk"],
        opar["-mut"], opar["-muw"],
        opar["-beta"], opar["-t1"], opar["-t2"],
        opar["-nmin"], opar["-nmax"],
        opar["-on"], opar["-om"],
        str(MAXI-1), __mode__, "discovery_7", __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, OVERLAPHRH):
      # Plotting ----
      print("Statistical analysis")
      plot_os = PLOT_OS_SF(data)
      # plot_os.plot_measurements_D(on=T)
      # plot_os.plot_measurements_D_noodle(on=T)
      # plot_os.plot_measurements_X(on=T)
      # plot_os.plot_measurements_X_noodle(on=T)
      # plot_os.plot_measurements_mu(on=T)
      # plot_os.plot_measurements_mu_noodle(on=T)
      # plot_os.plot_measurements_ntrees(on=T)
      # plot_os.plot_measurements_ntrees_noodle(on=T)
      # plot_os.plot_measurements_ordp(on=T)
      # plot_os.plot_measurements_ordp_noodle(on=T)
      # plot_os.ROC_OCN(on=T, hue_order=opt_score)
      # plot_os.histogram_overlap_scores(on=T, c=T, hue_order=opt_score)
      plot_os.histogram_clustering_similarity(
        on=T, c=T, hue_norm=[s.replace("_", "") for s in opt_score]
      )