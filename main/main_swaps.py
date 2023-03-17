# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Allias booleans ----
T = True
F = False
# Personal libs ----
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from various.data_transformations import maps
from networks.swapnet import SWAPNET
from various.network_tools import *
# Declare global variables ----
__iter__ = 0
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
run = T
__model__ = "1k"
structure = "LN"
distance = "tracto16"
topology = "MIX"
mapping = "R4"
index  = "simple"
__mode__ = "ALPHA"
imputation_method = ""
opt_score = ["_maxmu", "_X"]
__nodes__ = 57
__version__ = 220830
bias = float(0)
save_data = T

if __name__ == "__main__":
  # Create EDR network ----
  NET = SWAPNET(
    __nodes__,
    linkage,
    __mode__,
    __iter__,
    strcuture = structure,
    version=__version__,
    model=__model__,
    topology=topology,
    mapping=mapping, index=index,
    distance=distance,
    nlog10=nlog10, lookup=lookup,
    cut=cut, b=bias
  )
  NET.create_csv_path()
  # NET.create_pickle_path()
  # NET.create_plot_path()
  # Create network ----
  print("Create random graph")
  NET.random_one_k(run=run, on_save_csv=T)
  # Transform data for analysis ----
  R, lookup, shift = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  # Add color ----
  L = colregion(NET)
  NET.set_colregion(L)
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A[:, :NET.nodes], R[:, :NET.nodes],
      NET.D, __nodes__, linkage, __mode__, lookup=lookup
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    H.set_colregion(L)
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(H.subfolder),
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(NET.subfolder)
    )
  # Entropy ----
  HS = Hierarchical_Entropy(
    H.Z, H.nodes, H.colregion.labels[:H.nodes]
  )
  HS.Z2dict("short")
  HS.zdict2newick(HS.tree, weighted=F, on=F)
  HS.zdict2newick(HS.tree, weighted=T, on=F)
  node_entropy = HS.S(HS.tree)
  node_entropy_H = HS.S_height(HS.tree)
  # Plot H ----
  plot_h = Plot_H(NET, H)
  plot_h.Mu_plotly(on=F)
  plot_h.X_plotly(on=F)
  plot_h.D_plotly(on=F)
  # Plot N ----
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(NET.A, s=5, on=F)
  plot_n.A_vs_dis(NET.C, s=5, name="count", on=F)
  plot_n.histogram_weight(on=F)
  plot_n.projection_probability(
    NET.A[:, :__nodes__], bins=12, on=F
  )
  plot_n.plot_akis(
    NET.D, s=5, on=F
  )
  for score in opt_score:
    print(f"Find node partition using {score}")
    # Get best K and R ----
    k, r = get_best_kr_equivalence(score, H)
    print(
      "Best K: {}\nBest R: {}".format(k, r)
    )
    ## Take a look in case of SLN ----
    rlabels = get_labels_from_Z(H.Z, r)
    NET.overlap, _ = H.get_ocn_discovery(k, rlabels)
    H.set_overlap_labels(NET.overlap, score)
    ## Single linkage ----
    plot_h.core_dendrogram([r], on=F)
    plot_h.heatmap_pure(
      r, on=F, labels = rlabels,
      score="_"+score
    )
    plot_h.heatmap_dendro(
      r, on=F, score="_"+score
    )
    plot_h.lcmap_dendro(
      [k], on=F, score="_"+score
    )
  print("End!")
  # #@@ Todo: