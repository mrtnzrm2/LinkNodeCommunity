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
from networks.structure import STR
from various.network_tools import *
from various.fit_tools import fitters
from modules.discovery import discovery_channel
# Declare global variables ----
__iter__ = 1
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "FLNe"
distance = "MAP3D"
nature = "original"
__mode__ = "ZERO"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
imputation_method = ""
opt_score = ["_S"]
save_datas = T
# Declare global variables DISTBASE ----
__model__ = "M"
total_nodes = 91
__inj__ = 40
__nodes__ = 40
__bin__ = 12
__version__ = f"{__inj__}d{total_nodes}"
bias = 0.
alpha = 0.
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
  REF = STR[f"MAC{__inj__}"](
    linkage, __mode__,
    structure = structure,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology=topology,
    index=index,
    mapping=mapping,
    cut=cut, b=bias,
    discovery=discovery
  )
  _, _, _, _, est = fitters["EXPMLE"](REF.D, REF.CC, __bin__)
  lb = est.coef_[0]
  loc= est.loc
  # Create EDR network ----
  NET = DISTBASE(
    __inj__, total_nodes,
    linkage, __bin__, __mode__, __iter__,
    structure = structure,
    nlog10=nlog10, lookup=lookup, cut=cut,
    topology=topology, distance=distance,
    mapping=mapping, index=index, version = __version__,
    lb=lb, b=bias, model=__model__, discovery=discovery
  )
  NET.create_plot_path()
  # NET.create_csv_path()
  NET.create_pickle_path()
  L = colregion(REF, labels_name=f"labels{__inj__}")
  NET.set_labels(L.labels)
  # Create distance matrix ----
  D = REF.D
  # Create network ----
  print("Create random graph")
  Gn = NET.distbase_dict[__model__](
      D, REF.CC, loc=loc, run=T, on_save_csv=F
    )
  Ga = column_normalize(Gn)
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    Ga, nlog10, lookup, prob, b=bias
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_datas:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, Ga[:, :NET.nodes], R[:, :NET.nodes], D,
      __nodes__, linkage, __mode__,
      lookup=lookup, alpha=alpha, index=index
    )
    ## Compute quality functions ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    # Update entropy ----
    H.entropy = [
      H.node_entropy, H.node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    # Add labels ----
    H.set_colregion(L)
    H.delete_dist_matrix()
    save_class(
      H, NET.pickle_path,
      "hanalysis",
      on=F
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )  
  # Plot H ----
  plot_h = Plot_H(NET, H)
  HS = Hierarchical_Entropy(H.Z, H.nodes, H.colregion.labels[:H.nodes])
  HS.Z2dict("short")
  HS.zdict2newick(HS.tree, weighted=F, on=T)
  plot_h.plot_newick_R(HS.newick, HS.total_nodes, weighted=T, on=T)
  # HS.zdict2newick(HS.tree, weighted=T, on=T)
  # plot_h.plot_newick_R(HS.newick, weighted=T, on=T)
  # plot_h.plot_measurements_Entropy(on=T)
  # plot_h.plot_measurements_D(on=T)
  # plot_h.plot_measurements_S(on=T)
  # plot_h.plot_measurements_X(on=T)
  # Plot N ----

  RN = Ga[:__nodes__, :__nodes__].copy()
  RN[RN > 0] = -np.log(RN[RN > 0])
  np.fill_diagonal(RN, 0.)

  RW = Ga.copy()
  RW[RW > 0] = -np.log(RW[RW > 0])
  np.fill_diagonal(RW, 0.)

  RW10 = Ga.copy()
  RW10[RW10 > 0] = -np.log10(RW10[RW10 > 0])
  np.fill_diagonal(RW10, 0.)

  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(-RW, on=T)

  plot_n.histogram_weight(-RW10, on=T, label="logFLNe")
  # plot_n.histogram_dist(on=F)
  plot_n.projection_probability(
    Gn[:, :__nodes__], "EXPMLE", bins=__bin__, on=T
  )
  plot_n.plot_akis(D, s=7, on=T)
  for score in opt_score:
    K, R, HT = get_best_kr_equivalence(score, H)
    r = R[K == np.min(K)]
    k = K[K == np.min(K)]
    print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, score))
    rlabels = get_labels_from_Z(H.Z, r)
    rlabels = skim_partition(rlabels)
    # Overlap ----
    for direction in ["source", "target", "both"]:
      print("***", direction)
      ocn, data_nocs, noc_sizes, rlabels2 = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)
      NET.set_overlap(ocn)
      cover = omega_index_format(rlabels2,  data_nocs, NET.struct_labels[:NET.nodes])
      H.set_cover(cover, score, direction)
      H.set_rlabels(rlabels2, score, direction)
      H.set_overlap_labels(NET.overlap, score, direction)
    # Plot N ----

      plot_n.plot_network_covers(
        k, RN, rlabels2, rlabels,
        data_nocs, noc_sizes, H.colregion.labels[:H.nodes],
        score=score, direction=direction, cmap_name="hls", on=T, figsize=(8,8)
      )
    ## Single linkage ----
    plot_h.heatmap_dendro(r, -RN, on=T, score="logFLNe")
    plot_h.lcmap_dendro(k, r, on=T, font_size = 12)
    plot_h.flatmap_dendro_91(
      NET, H, cmap_name="hls" #
    )
  # save_class(
  #   H, NET.pickle_path,
  #   "hanalysis",
  #   on=T
  # )
  print("End!")