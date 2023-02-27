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
from networks_serial.scalehrh import SCALEHRH
from plotting_modules.plotting_serial_sf import PLOT_S_SF
from networks.scalefree import SCALEFREE
from various.network_tools import *
from numpy import zeros
# Declare global variables NET ----
MAXI = 500
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
topology = "MIX"
mapping = "trivial"
index = "jacp"
__mode__ = "ALPHA"
opt_score = ["_maxmu", "_X", "_D"]
save_data = F
save_hierarchy = F
# WDN paramters ----
par = {
  "-N" : "{}".format(str(__nodes__)),
  "-k" : "7.0",
  "-maxk" : "50.0",
  "-mut" : "0.2",
  "-muw" : "0.2",
  "-beta" : "2.5",
  "-t1" : "2",
  "-t2" : "1",
  "-nmin" : "5",
  "-nmax" : "20"
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
if __name__ == "__main__":
  # Store data ---
  data = SCALEHRH(linkage)
  RAND_H = 0
  rlabels = 0
  # RANDOM networks ----
  print("Create random networks ----")
  if save_data:
    i = 0
    while i < MAXI:
      print("***\tIteration: {}".format(i))
      # Add iteartion to data ----
      data.set_iter(i)
      RAND = SCALEFREE(
        i,
        linkage,
        __mode__,
        nlog10=nlog10, lookup=lookup,
        cut=cut,
        topology=topology,
        mapping=mapping, index=index,
        parameters=par
      )
      # Create network ----
      print("Create random graph")
      RAND.random_WDN_cpp(run=run, on_save_pickle=F)     #*****
      RAND.col_normalized_adj(on=F)
      # Compute RAND Hierarchy ----
      print("Compute Hierarchy")
      if save_hierarchy:
        RAND_H = Hierarchy(
          RAND, RAND.A, RAND.A, zeros(RAND.A.shape),
          __nodes__, linkage, __mode__
        )
        ## Compute features ----
        RAND_H.BH_features_cpp()
        ## Compute lq arbre de merde ----
        RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
        # Save stats ----
        data.set_data_measurements(RAND_H, i)
        save_class(
          RAND_H, RAND.pickle_path,
          "hanalysis_{}".format(RAND_H.subfolder),
          on=F
        )
      else:
        RAND_H = read_class(
          RAND.pickle_path,
          "hanalysis_{}".format(RAND.subfolder),
        )
      for score in opt_score:
        print("Score: {}".format(score))
        # Get best k, r for given score ----
        k, r = get_best_kr(score, RAND_H)
        # Single linkage part ----
        print("Best K: {}\nBest R: {}".format(k, r))
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        if np.nan in rlabels:
          print("*** BAD dendrogram")
          break
        data.set_nmi_nc(RAND.labels, rlabels,score = score)
      if np.sum(np.isnan(rlabels)) == 0: i += 1
    # Save ----
    if isinstance(RAND_H, Hierarchy):
      data.set_subfolder(RAND_H.subfolder)
      data.set_plot_path(RAND_H) 
      data.set_pickle_path(RAND_H)
      print("Save data")
      save_class(
        data, data.pickle_path,
        "series_{}".format(MAXI)
      )
  else:
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/{}{}{}{}/{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        linkage.upper(), l10, lup, _cut,
        __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
  # Plotting ----
  print("Statistical analysis")
  plot_s = PLOT_S_SF(data)
  plot_s.plot_measurements_D(on=T)
  plot_s.plot_measurements_D_noodle(on=T)
  plot_s.plot_measurements_X(on=T)
  plot_s.plot_measurements_X_noodle(on=T)
  plot_s.plot_measurements_mu(on=T)
  plot_s.plot_measurements_mu_noodle(on=T)
  plot_s.plot_measurements_ntrees(on=T)
  plot_s.plot_measurements_ntrees_noodle(on=T)
  plot_s.plot_measurements_ordp(on=T)
  plot_s.plot_measurements_ordp_noodle(on=T)
  plot_s.histogram_nmi(on=T, c=T, hue_norm=opt_score)