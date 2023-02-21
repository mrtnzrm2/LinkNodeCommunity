# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Import libraries ----
## Standard libs ----
from numpy import zeros, arange
## Personal libs ----
from modules.hierarmerge import Hierarchy
from networks_serial.overlaphrh import OVERLAPHRH
from plotting_modules.plotting_o_serial_sf import PLOT_OS_SF
from networks.overlapping import OVERLAPPING
from modules.colregion import colregion
from various.network_tools import *
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
# Overlapping WDN paramters ----
opar = {
  "-N" : "{}".format(
    str(__nodes__)
  ),
  "-k" : "7.0",
  "-maxk" : "100",
  "-mut" : "0.2",
  "-muw" : "0.2",
  "-beta" : "2.5",
  "-t1" : "2.5",
  "-t2" : "2.5",
  "-on" : "10",
  "-om" : "2"
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
# Start main ----
if __name__ == "__main__":
  # Store data ---
  data = OVERLAPHRH(linkage)
  RAND_H = 0
  # RANDOM networks ----
  serie = arange(MAXI)
  print("Create random networks ----")
  if save_data:  
    for i in serie:
      print("***\tIteration: {}".format(i))
      # Set iter to data ----
      data.set_iter(i)
      RAND = OVERLAPPING(
        i,
        linkage,
        __mode__,
        nlog10=nlog10, lookup=lookup,
        topology=topology,
        mapping=mapping, index=index,
        cut=cut,
        parameters=opar
      )
      # Create network ----
      print("Create random graph")
      RAND.random_WDN_overlap_cpp(
        run=run, on_save_pickle = F   #****
      )
      RAND.col_normalized_adj(on=F)
      L = colregion(RAND)
      RAND.set_colregion(L)
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
        RAND_H.set_colregion(L)
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
        # Get best k, r for given score ----
        k, r = get_best_kr_equivalence(score, RAND_H)
        # Single linkage part ----
        print(
          "Best K: {}\nBest R: {}".format(k, r)
        )
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        # NMI between ground-truth and pred labels ----
        data.set_nmi_nc_overlap(
          RAND.labels, rlabels, RAND.overlap,
          score = score
        )
        # Accuracy and false-positive ratio of
        # multicommunity nodes ----
        # sen, sep = RAND.overlap_score(RAND_H, [k], rlabels, on=T)
        sen, sep = RAND.overlap_score_discovery(RAND_H, k, rlabels, on=T)
        data.set_overlap_scores(
          sen, sep, score = score
        )
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
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-on_{}/-om_{}/{}{}{}{}/{}/{}".format(
        str(__nodes__),
        opar["-k"], opar["-maxk"],
        opar["-mut"], opar["-muw"],
        opar["-beta"], opar["-t1"], opar["-t2"],
        opar["-on"], opar["-om"],
        linkage.upper(), l10, lup, _cut,
        __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
  # Plotting ----
  print("Statistical analysis")
  plot_os = PLOT_OS_SF(data)
  plot_os.plot_measurements_D(on=T)
  plot_os.plot_measurements_D_noodle(on=T)
  plot_os.plot_measurements_X(on=T)
  plot_os.plot_measurements_X_noodle(on=T)
  plot_os.plot_measurements_mu(on=T)
  plot_os.plot_measurements_mu_noodle(on=T)
  plot_os.plot_measurements_ntrees(on=T)
  plot_os.plot_measurements_ntrees_noodle(on=T)
  plot_os.plot_measurements_ordp(on=T)
  plot_os.plot_measurements_ordp_noodle(on=T)
  plot_os.ROC_OCN(on=T, hue_norm=opt_score)
  plot_os.histogram_overlap_scores(on=T, c=T, hue_norm=opt_score)
  plot_os.histogram_nmi(on=T, c=T, hue_norm=opt_score)