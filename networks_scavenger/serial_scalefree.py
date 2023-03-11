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
import itertools
# Personal libs ----
from networks_serial.scalehrh import SCALEHRH
from plotting_modules.plotting_serial_sf import PLOT_S_SF
from various.network_tools import read_class
# Declare iter variables ----
topologies = ["TARGET", "SOURCE", "MIX"]
indices = ["jacw", "jacp", "cos", "bsim"]
KAV = [15]
MUT = [0.1, 0.3, 0.5]
MUW = [0.1, 0.5]
list_of_lists = itertools.product(
  *[topologies, indices, KAV, MUT, MUW]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 503
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "ALPHA"
opt_score = ["_maxmu", "_X", "_D"]
# Start main ----
if __name__ == "__main__":
  for topology, index, kav, mut, muw in list_of_lists:
    # WDN paramters ----
    par = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{kav}.0",
      "-maxk" : "30",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2.5",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "2",
      "-nmax" : "10"

    }
    # Print summary ----
    print("Number of iterations:")
    print(MAXI)
    print("For NET parameters:")
    print(
      "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup: {}".format(
        linkage, __mode__, opt_score, nlog10, lookup
      )
    )
    print("WDN parameters:")
    print(par)
    # Start main ----
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}{}{}{}/{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"], MAXI,
        linkage.upper(), l10, lup, _cut,
        __mode__,
        f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, SCALEHRH):
      # Plotting ----
      print("Statistical analysis")
      plot_s = PLOT_S_SF(data)
      plot_s.plot_measurements_D(on=T)
      plot_s.plot_measurements_D_noodle(on=T)
      # plot_s.plot_measurements_X(on=T)
      # plot_s.plot_measurements_X_noodle(on=T)
      # plot_s.plot_measurements_mu(on=T)
      # plot_s.plot_measurements_mu_noodle(on=T)
      # plot_s.plot_measurements_ntrees(on=T)
      # plot_s.plot_measurements_ntrees_noodle(on=T)
      # plot_s.plot_measurements_ordp(on=T)
      # plot_s.plot_measurements_ordp_noodle(on=T)
      if par["-mut"] == "0.1" and par["-muw"] == "0.5" and par["-k"] == "15.0": continue
      plot_s.histogram_clustering_similarity_old(
        on=T, c=T, hue_norm=[s.replace("_", "") for s in opt_score]
      )