# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
# Personal libraries ----
import ctools as ct
from networks.structure import STR
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject = "MUS"
structure = "FLN"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
version = "19"+"d"+"47"
__nodes__ = 19
__inj__ = 19

if __name__ == "__main__":
    NET = STR[f"{subject}{__inj__}"](
      linkage, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      discovery = discovery,
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = alpha
    )

    NET_H = read_class(
      NET.pickle_path,
      "hanalysis"
    )

    nodes = NET.nodes

    mean_out_distinguishability_p = {r : [] for r in np.arange(nodes-1, 0, -1)}
    mean_in_distinguishability_p = {r : [] for r in np.arange(nodes-1, 0, -1)}
    mean_out_distinguishability_n = {r : [] for r in np.arange(nodes-1, 0, -1)}
    mean_in_distinguishability_n = {r : [] for r in np.arange(nodes-1, 0, -1)}

    sm = -2 * np.log(NET_H.source_sim_matrix)
    tm = -2 * np.log(NET_H.target_sim_matrix)
    sm[sm == np.Inf] = np.nan
    tm[tm == np.Inf] = np.nan

    # e = 0
    for r in np.arange(nodes - 1, 0, -1):
        labels = get_labels_from_Z(NET_H.Z, r)
        # labels = np.array(skim_partition(labels))
        ulabels = np.unique(labels)
        ulabels = np.array([nm for nm in ulabels if nm != -1])
        for i in np.arange(ulabels.shape[0]):
            for j in np.arange(i, ulabels.shape[0]):
              comm_i_nodes = np.where(labels == ulabels[i])[0]
              comm_j_nodes = np.where(labels == ulabels[j])[0]
              if i == j:
                  m = np.nanmean(sm[comm_i_nodes, :][:, comm_j_nodes])
                  if not np.isnan(m) and m < np.Inf:
                    mean_in_distinguishability_p[r].append(m)
                  m = np.nanmean(tm[comm_i_nodes, :][:, comm_j_nodes])
                  if not np.isnan(m) and m < np.Inf:
                    mean_in_distinguishability_n[r].append(m)
              else:
                  m = np.nanmean(sm[comm_i_nodes, :][:, comm_j_nodes])
                  if not np.isnan(m) and m < np.Inf:
                    mean_out_distinguishability_p[r].append(m)
                  m = np.nanmean(tm[comm_i_nodes, :][:, comm_j_nodes])
                  if not np.isnan(m) and m < np.Inf:
                    mean_out_distinguishability_n[r].append(m)

    array_mean_in_distinguishability_p = np.zeros(nodes - 1)
    array_std_in_distinguishability_p = np.zeros(nodes - 1)
    array_mean_in_distinguishability_n = np.zeros(nodes - 1)
    array_std_in_distinguishability_n = np.zeros(nodes - 1)

    array_mean_out_distinguishability_p = np.zeros(nodes - 1)
    array_std_out_distinguishability_p = np.zeros(nodes - 1)
    array_mean_out_distinguishability_n = np.zeros(nodes - 1)
    array_std_out_distinguishability_n = np.zeros(nodes - 1)

    for i in np.arange(nodes - 1, 0, -1):
       array_mean_in_distinguishability_p[i-1] = np.nanmean(mean_in_distinguishability_p[i])
       array_mean_out_distinguishability_p[i-1] = np.nanmean(mean_out_distinguishability_p[i])

       array_std_in_distinguishability_p[i-1] = np.nanstd(mean_in_distinguishability_p[i])
       array_std_out_distinguishability_p[i-1] = np.nanstd(mean_out_distinguishability_p[i])

       array_mean_in_distinguishability_n[i-1] = np.nanmean(mean_in_distinguishability_n[i])
       array_mean_out_distinguishability_n[i-1] = np.nanmean(mean_out_distinguishability_n[i])

       array_std_in_distinguishability_n[i-1] = np.nanstd(mean_in_distinguishability_n[i])
       array_std_out_distinguishability_n[i-1] = np.nanstd(mean_out_distinguishability_n[i])

    data = pd.DataFrame(
      {
          "Info" : list(array_mean_out_distinguishability_n - array_mean_in_distinguishability_n) + list(array_mean_out_distinguishability_p - array_mean_in_distinguishability_p),
          "R" : list(np.arange(1, nodes)) * 2,
          "set" : ['in'] * (nodes-1) + ['out'] * (nodes-1)
      }
    )

    sns.scatterplot(
       data=data,
       x="R",
       y="Info",
       hue="set"
    )

    # import matplotlib.lines as mlines

    # for i in np.arange(nodes - 1):
    #   l = mlines.Line2D(
    #      [i, i], [
    #       array_mean_in_distinguishability_n[i] - array_std_in_distinguishability_n[i]/2,
    #       array_mean_in_distinguishability_n[i] + array_std_in_distinguishability_n[i]/2
    #     ],
    #     color=sns.color_palette("deep")[0]
    #   )
    #   plt.gca().add_line(l)
    #   l = mlines.Line2D(
    #      [i, i], [
    #       array_mean_out_distinguishability_n[i] - array_std_out_distinguishability_n[i]/2,
    #       array_mean_out_distinguishability_n[i] + array_std_out_distinguishability_n[i]/2
    #     ],
    #     color=sns.color_palette("deep")[1]
    #   )
    #   plt.gca().add_line(l)

    

    plt.show()

