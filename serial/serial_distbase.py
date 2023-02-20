# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libraries ----
from numpy import arange
# Import network libraries ----
from modules.hierarmerge import Hierarchy
from networks_serial.hrh import HRH
from various.network_tools import *
from modules.colregion import colregion
from plotting_modules.plotting_serial import PLOT_S
from plotting_modules.plotting_o_serial import PLOT_OS
from networks.structure import MAC
from networks.distbase import DISTBASE
# Declare global variables NET ----
MAXI = 502
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
run = T
distance = "MAP3D"
nature = "original"
mode = "ALPHA"
topology = "SOURCE"
mapping = "R2"
index  = "jacw"
imputation_method = ""
opt_score = ["_maxmu", "_X", "_D"]
save_data = F
save_hierarchy = F
# Statistic test ----
alternative = "less"
# Declare global variables DISTBASE ----
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__version__ = 220830
__model__ = "DEN"
__bin__ = 12
bias = 1e-5
# Print summary ----
print("For NET parameters:")
print(
  "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\nlookup : {}".format(
    linkage, mode, opt_score, nlog10, lookup
  )
)
print("For imputation parameters:")
print(
  "nature: {}\nmodel: {}".format(
    nature, imputation_method
  )
)
print("Random network and statistical paramteres:")
print(
  "distbase: {}\nnodes: {}\ninj: {}\nalternative: {}".format(
    __model__, str(__nodes__),str(__inj__), alternative
  )
)
# Start main ----
if __name__ == "__main__":
  print("Load MAC data ----")
  # Create macaque class ----
  NET = MAC(
    linkage, mode,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    nature=nature,
    model=imputation_method,
    distance=distance,
    topology=topology, index=index,
    inj=__inj__,
    mapping=mapping,
    cut=cut, b=bias
  )
  # Load hierarhical analysis ----
  NET_H = read_class(
    NET.pickle_path,
    "hanalysis_{}".format(NET.subfolder),
  )
  # Create colregions ----
  L = colregion(NET)
  # Create hrh class ----
  data = HRH(NET_H, L)
  for score in opt_score:
    overlap_labels = NET_H.overlap.labels.loc[
      NET_H.overlap.score == score
    ].to_numpy()
    data.set_overlap_data_one(overlap_labels, score)
    data.set_nodes_labels_single(NET_H, score)
  RAND_H = 0
  # RANDOM networks ----
  serie = arange(MAXI)
  print("Create random networks ----")
  if save_data:  
    for i in serie:
      print("Iteration: {}".format(i))
      # Add iteartion to data ----
      data.set_iter(i)
      RAND = DISTBASE(
        __inj__, total_nodes,
        linkage, __bin__, mode, i,
        version=__version__, model=__model__,
        nlog10=nlog10, lookup=lookup, cut=cut,
        topology=topology, distance=distance,
        index=index, mapping=mapping,
        lb=0.07921125, b=bias
      )
      RAND.create_pickle_path()
      # Create distance matrix ----
      D = RAND.get_distance_matrix(NET.struct_labels)
      # Create network ----
      print("Create random graph")
      RC = RAND.distbase_dict[__model__](
        D, NET.C, run=run, on_save_csv=F
      )
      R = column_normalize(RC)
      # Compute RAND Hierarchy ----
      print("Compute Hierarchy")
      if save_hierarchy:
        RAND_H = Hierarchy(
          RAND, R[:, :__nodes__], D,
          __nodes__, linkage, mode,
          prob=prob, b=bias
        )
        ## Compute features ----
        RAND_H.BH_features_cpp()
        ## Compute lq arbre de merde ----
        RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
        RAND_H.set_colregion(L)
        # Saving statistics ----
        data.set_data_homogeneity_zero(RAND_H.R)
        data.set_data_measurements_zero(RAND_H, i)
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
      # Convergence ----
      data.set_stats(RAND_H)
      for score in opt_score:
        # Get k from RAND_H ----
        k, r = get_best_kr_equivalence(score, RAND_H)
        RAND_H.set_kr(k, r, score)
        data.set_kr_zero(RAND_H)
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        data.set_nmi_nc(rlabels[:__inj__], score)
        # Overlap ----
        # ocn = RAND_H.get_ocn(rlabels)
        ocn, _ = RAND_H.get_ocn_discovery(k, rlabels) ## change <-----
        data.set_overlap_data_zero(ocn, score)
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
    data = read_class(
      "../pickle/RAN/distbase/MAC/{}/FLN/{}/{}/BIN_{}/{}/{}/{}".format(
        __version__,
        distance,
        __model__,
        str(__bin__),
        NET.analysis,
        mode,
        NET.subfolder
      ),
      "series_{}".format(MAXI)
    )
  # Plotting ----
  print("Statistical analysis")
  plot_s = PLOT_S(data)
  plot_s.scatterplot_NH(on=T)
  plot_s.plot_stats(
    alternative=alternative,
    on=T,
  )
  plot_s.plot_measurements_D(on=T)
  plot_s.plot_measurements_X(on=T)
  plot_s.plot_measurements_mu(on=T)
  plot_s.plot_measurements_ntrees(on=T)
  plot_s.plot_measurements_ordp(on=T)
  plot_s.plot_measurements_D_noodle(on=T)
  plot_s.plot_measurements_X_noodle(on=T)
  plot_s.plot_measurements_mu_noodle(on=T)
  plot_s.plot_measurements_ntrees_noodle(on=T)
  plot_s.plot_measurements_ordp_noodle(on=T)
  plot_o = PLOT_OS(data)
  for score in opt_score:
    plot_s.histogram_krs(score=score, on=T)
    plot_s.histogram_nmi(score, on=T)
    plot_o.histogram_overlap(score, on=T)