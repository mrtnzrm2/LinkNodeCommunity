# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
## My libraries ----
from networks.structure import MAC
from modules.nodhierarchy import NODH
from plotting_modules.plotting_trends import PLOT_TREND
from plotting_modules.plotting_H import Plot_H
# Boolean aliases ----
T = True
F = False

if __name__ == "__main__":
  linkage = "single"
  mode = "ALPHA"
  version = 220830
  nature = "original"
  distance = "MAP3D"
  imputation_model = ""
  direct = "SOURCE_FULL"
  feature = "dtw_source"
  nodes=106
  inj=57
  nlog10=T
  lookup=F
  
  ## Creat monkey ----
  NET = MAC(
    linkage, mode, nlog10=nlog10, lookup=lookup,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature,
    inj=inj
  )
  # Create Hierarchy ----
  H = NODH(NET, nodes, nlog10=nlog10, lookup=lookup)
  ##
  H.hierarchical_clustering()
  # cluster, i = H.best_partition_similarity(save=T)
  cluster, i = H.best_partition_X(save=T)
  rlabel = H.cut_tree(cluster)
  # Create plots ----
  plot_trend = PLOT_TREND(NET, H)
  plot_trend.plot_mean_trend_histo([cluster], H.Z, axis=0, on=T)
  plot_trend.plot_mean_trend_hist_area([cluster], H.Z, axis=0, on=T)
  plot_trend.plot_flatmap_220830(
    [cluster], H.Z, on=T,
    # EC=T
  )
  plot_trend.plot_dendrogram(H.feature_dist ,NET.struct_labels, on=F)
  plot_trend.plot_feature_dist_dist(cluster, on=T)
  plot_trend.plot_wtrends_dendrogram([cluster], NET.struct_labels, H.Z, axis=0, on=T)
  plot_h = Plot_H(NET, H)
  plot_h.heatmap_pure(cluster, labels=rlabel, on=F)
  plot_h.core_dendrogram([cluster], on=T)
  ## In case of source and target similarity comparison
  # plot_trend = PLOT_TREND(NET, H)
  # plot_trend.plot_dtw_both(NET.struct_labels)
  # plot_trend.plot_winout_trends(NET.struct_labels)

