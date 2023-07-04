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
import numpy as np
# Personal libs ----
from plotting_modules.plotting_serial import PLOT_S
from plotting_modules.plotting_o_serial import PLOT_OS
from various.network_tools import read_class
# Declare iter variables ----
topologies = ["MIX"]
bias = [0]
mode = ["ZERO"]
list_of_lists = itertools.product(
  *[topologies, bias, mode]
)
list_of_lists = np.array(list(list_of_lists))
# Declare global variables NET ----
MAXI = 500
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
run = T
structure = "LN"
distance = "tracto16"
nature = "original"
mapping = "trivial"
index = "D1_2_3"
imputation_method = ""
opt_score = ["_maxmu_6", "_maxmu_20", "_X"]
# Declare global variables DISTBASE ----
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__version__ = "57d106"
__model__ = "1k"
# T test ----
alternative = "less"
# Print summary ----
print("For NET parameters:")
print(
  "linkage: {}\nscore: {}\nnlog: {}\n lookup: {}".format(
    linkage, opt_score, nlog10, lookup
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
  for topology, bias, mode in list_of_lists:
    bias = float(bias)
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    print("Load MAC data ----")
    data = read_class(
      "../pickle/RAN/swaps/MAC/{}/{}/{}/{}/{}/{}/{}/b_{}".format(
        __version__,
        structure,
        distance,
        __model__,
       f"{linkage.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
        mode,
        f"{topology}_{index}_{mapping}",
        bias
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, int): continue
    # Plotting ----
    print("Statistical analysis")
    plot_s = PLOT_S(data)
    plot_s.plot_measurements_Entropy(on=T)
    plot_s.plot_measurements_Entropy_noodle(on=F)
    plot_s.plot_stats(alternative=alternative, on=T)
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
    plot_s.histogram_clustering_similarity(
      on=T, c=T, hue_norm=[s.replace("_", "") for s in opt_score]
    )
    plot_o = PLOT_OS(data)
    for score in opt_score:
      plot_s.histogram_krs(score=score, on=T)
      plot_o.histogram_overlap(score, on=T)
    
  