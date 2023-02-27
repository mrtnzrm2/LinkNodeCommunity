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
from modules.colregion import colregion
from networks_serial.hrh import HRH
from plotting_modules.plotting_serial import PLOT_S
from plotting_modules.plotting_H import Plot_H
from networks.structure import MAC
from networks.swapnet import SWAPNET
from various.data_transformations import maps
from various.network_tools import *
# Declare global variables NET ----
MAXI = 3
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
run = T
distance = "MAP3D"
nature = "original"
topology = "MIX"
mapping = "R2"
index = "jacw"
mode = "ALPHA"
imputation_method = ""
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
save_hierarchy = T
# Declare global variables DISTBASE ----
__inj__ = 57
__nodes__ = 57
__version__ = 220830
__model__ = "1k"
bias = 0.3
# T test ----
alternative = "less"
# Print summary ----
print("For NET parameters:")
print(
  "linkage: {}\nmode: {}\nscore: {}\nnlog: {}\n lookup: {}".format(
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
  "nodes: {}\ninj: {}\nalternative: {}".format(
    str(__nodes__),str(__inj__), alternative
  )
)
# Start main ----
if __name__ == "__main__":
  print("Load MAC data ----")
  # Create macaque class ----
  NET = MAC(
    linkage, mode, nlog10, lookup,
    version = __version__,
    distance = distance,
    nature = nature,
    model = imputation_method,
    inj = __inj__,
    topology = topology,
    mapping=mapping, index=index,
    cut = cut, b=bias
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
  RAND_H = 0
  for score in opt_score:
    k, r = get_best_kr(score, NET_H)
    rlabels = get_labels_from_Z(NET_H.Z, r)
    overlap_labels = NET_H.overlap.labels.loc[
      NET_H.overlap.score == score
    ].to_numpy()
    data.set_overlap_data_one(overlap_labels, score)
    data.set_nodes_labels(rlabels, score)
    # Cover
    data.set_cover_one(NET_H.cover[score], score)
  # RANDOM networks ----
  from numpy import arange
  serie = arange(MAXI)
  print("Create random networks ----")
  if save_data:  
    for i in serie:
      print("***\tIteration: {}".format(i))
      data.set_iter(i)
      RAND = SWAPNET(
        __inj__, linkage,
        mode, i,
        topology=topology,
        nature = nature,
        mapping=mapping, index=index,
        distance = distance,
        nlog10 = nlog10, lookup = lookup,
        version=__version__, model=imputation_method,
        cut=cut, b=bias
      )
      # Create network ----
      print("Create random graph")
      RAND.random_one_k(run=run, on_save_csv=F)   #****
      # Transform data for analysis ----
      R, lookup, _ = maps[mapping](
        RAND.A, nlog10, lookup, prob, b=bias
      )
      # Compute RAND Hierarchy ----
      print("Compute Hierarchy")
      if save_hierarchy:
        RAND_H = Hierarchy(
          RAND, RAND.A[:, :__nodes__], R[:, :__nodes__], RAND.D,
          __nodes__, linkage, mode, lookup=lookup
        )
        ## Compute features ----
        RAND_H.BH_features_cpp()
        ## Compute lq arbre de merde ----
        RAND_H.la_abre_a_merde_cpp(RAND_H.BH[0])
        RAND_H.set_colregion(L)
        # Saving statistics ----
        data.set_data_homogeneity_zero(RAND_H.R)
        data.set_data_measurements_zero(RAND_H, i)
        data.set_stats(RAND_H)
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
      # Plot ----
      plot_h = Plot_H(RAND, RAND_H)
      plot_h.Mu_plotly(on=F)
      plot_h.X_plotly(on=F)
      plot_h.D_plotly(on=F)
      for score in opt_score:
        # Get best k, r for given score ----
        k, r = get_best_kr(score, RAND_H)
        RAND_H.set_kr(k, r, score)
        data.set_kr_zero(RAND_H)
        # Add iteartion to data----
        rlabels = get_labels_from_Z(RAND_H.Z, r)
        plot_h.core_dendrogram([r], on=F)
        # Overlap ----
        ocn, subcover = RAND_H.get_ocn_discovery(k, rlabels)
        cover = omega_index_format(
          rlabels, subcover, RAND_H.colregion.labels[:RAND_H.nodes]
        )
        data.set_clustering_similarity(rlabels, cover, score)
        data.set_overlap_data_zero(ocn, score)
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
    data = read_class(
      "../pickle/RAN/swaps/MAC/{}/FLN/{}/1k/{}/{}/{}".format(
        __version__,
        distance,
        NET.analysis,
        mode,
        NET.subfolder
      ),
      "series_{}".format(MAXI)
    )
  # Plotting ----
  print("Statistical analysis")
  plot_s = PLOT_S(data)
  plot_s.plot_stats(
    alternative=alternative,
    on=T
  )
  plot_s.plot_measurements_D_noodle(on=T)
  plot_s.plot_measurements_X_noodle(on=T)
  plot_s.plot_measurements_mu_noodle(on=T)
  plot_s.plot_measurements_ntrees_noodle(on=T)
  plot_s.plot_measurements_ordp_noodle(on=T)
  plot_s.plot_measurements_D(on=T)
  plot_s.plot_measurements_X(on=T)
  plot_s.plot_measurements_mu(on=T)
  plot_s.plot_measurements_ntrees(on=T)
  plot_s.plot_measurements_ordp(on=T)
  for score in opt_score:
    plot_s.histogram_krs(score=score, on=T)
    plot_s.histogram_clustering_similarity(score, on=T)
  