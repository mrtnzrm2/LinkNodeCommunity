# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from various.data_transformations import maps
from networks.distbase import DISTBASE
from networks.structure import MAC
from various.network_tools import *
# Declare global variables ----
__iter__ = 0
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
distance = "MAP3D"
nature = "original"
__mode__ = "ALPHA"
topology = "MIX"
mapping = "R2"
index  = "jacw"
imputation_method = ""
opt_score = ["_maxmu", "_X"]
save_datas = T
# Declare global variables DISTBASE ----
__model__ = "DEN"
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__bin__ = 12
lb = 0.07921125
__version__ = 220830
bias = 0.3
## Very specific!!! Be careful ----
if nature == "original":
  __ex_name__ = f"{total_nodes}_{__inj__}"
else:
  __ex_name__ = f"{total_nodes}_{total_nodes}_{__inj__}"
if nlog10: __ex_name__ = f"{__ex_name__}_l10"
if lookup: __ex_name__ = f"{__ex_name__}_lup"
if cut: __ex_name__ = f"{__ex_name__}_cut"

if __name__ == "__main__":
  # MAC network as reference ----
  REF = MAC(
    linkage, __mode__,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology=topology,
    index=index,
    mapping=mapping,
    cut=cut, b=bias
  )
  # Create EDR network ----
  NET = DISTBASE(
    __inj__, total_nodes,
    linkage, __bin__, __mode__, __iter__,
    nlog10=nlog10, lookup=lookup, cut=cut,
    topology=topology, distance=distance,
    mapping=mapping, index=index, version = __version__,
    lb=lb, b=bias, model=__model__
  )
  # NET.create_plot_path()
  NET.create_csv_path()
  # NET.create_pickle_path()
  L = colregion(NET)
  NET.set_labels(L.labels)
  # Create distance matrix ----
  D = NET.get_distance_matrix(NET.struct_labels)
  # Create network ----
  print("Create random graph")
  Gn = NET.distbase_dict[__model__](
      D, REF.C, run=T, on_save_csv=T
    )
  G = column_normalize(Gn)
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    G, nlog10, lookup, prob, b=bias
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_datas:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, G[:, :NET.nodes], R[:, :NET.nodes], D,
      __nodes__, linkage, __mode__,
      lookup=lookup
    )
    ## Compute quality functions ----
    H.BH_features_cpp()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Add labels ----
    H.set_colregion(L)
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(NET.subfolder),
    )
  # Entropy ----
  HS = Hierarchical_Entropy(H.Z, H.nodes)
  HS.Z2dict("short")
  _, sv, sh = HS.S(HS.tree)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_h.Mu_plotly(on=F)
  plot_h.X_plotly(on=F)
  plot_h.D_plotly(on=F)
  # Plot N ----
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(G[:, :__nodes__], on=F)
  plot_n.A_vs_dis(Gn[:, :__nodes__], name="count", on=F)
  plot_n.histogram_weight(on=F)
  plot_n.histogram_dist(on=F)
  plot_n.projection_probability(
    Gn[:, :__nodes__], bins=12, on=F
  )
  plot_n.plot_akis(D, s=5, on=F)
  for score in opt_score:
    # Get best K and R ----
    k, r = get_best_kr(score, H)
    print("Best K: {}\nBest R: {}".format(k, r))
    rlabels = get_labels_from_Z(H.Z, r)
    # Overlap ----
    ocn, _ = H.get_ocn_discovery(k, rlabels)
    NET.set_overlap(ocn)
    H.set_overlap_labels(ocn, score)
    plot_h.core_dendrogram([r], on=F)
    ## Single linkage ----
    plot_h.heatmap_dendro(r, on=F)
    plot_h.lcmap_dendro([k], on=F)
    plot_h.flatmap_dendro(
      NET, [k], [r], on=F, EC=T #
    )
  save_class(
    H, NET.pickle_path,
    "hanalysis_{}".format(H.subfolder),
    on=F
  )
  print("End!")