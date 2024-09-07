# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Personal libs ---- 
from networks.MAC.mac49 import MAC49
from modules.hierarmerge import Hierarchy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.data_transformations import maps
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "49d106"
__nodes__ = 49
__inj__ = 49
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC49(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  ## Hierarchy object!! ----
  H = Hierarchy(
    NET, NET.A, R, NET.D,
    __nodes__, linkage, mode, lookup=lookup, index=index
  )

  D12 = H.target_sim_matrix
  D12[D12 != 0] = -2 * np.log(D12[D12 != 0])

  D12p = H.source_sim_matrix
  D12p[D12p != 0] = -2 * np.log(D12p[D12p != 0])

  SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

  A = NET.A[:NET.nodes, :][:, :NET.nodes]
  non_zero = A > 0

  # data = pd.DataFrame(
  #    {
  #       "FLNe" : list(np.log10(A[non_zero])) * 2,
  #       "SLN" : list(SLN[non_zero]) * 2,
  #       "D12" : np.hstack([D12[non_zero], D12p[non_zero]]),
  #       "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
  #    }
  # )

  data = pd.DataFrame(
     {
        "D" : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        "FLNe" : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        "D12" : D12[non_zero]
        # "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
     }
  )

  sns.set_style("ticks")

  true_bins = 15
  bins = true_bins - 1
  min_x = np.min(data["SLN"])
  max_x = np.max(data["SLN"])
  x_bin_boundaries = np.linspace(min_x, max_x, bins+1)
  bin_width = x_bin_boundaries[1]-x_bin_boundaries[0]
  x_bin_center = x_bin_boundaries[1:] - bin_width
  x_bin_center = np.hstack([x_bin_center, np.array(x_bin_center[-1] + bin_width)])
  x_bin_boundaries -= bin_width / 2
  x_bin_boundaries = np.hstack([x_bin_boundaries, np.array(x_bin_boundaries[-1] + bin_width)])
  # x_bin_boundaries[-1] += 1e-4
  y_average_center = np.zeros(x_bin_center.shape[0])
  y_std_center = np.zeros(x_bin_center.shape[0])


  xlabel = "FLNe"

  for i in np.arange(true_bins):
     y_average_center[i] = data[xlabel].loc[
          #  (data["class"] == "-") &
           (data["SLN"] >= x_bin_boundaries[i]) &
           (data["SLN"] < x_bin_boundaries[i+1])
        ].mean()
     y_std_center[i] = data[xlabel].loc[
          #  (data["class"] == "-") &
           (data["SLN"] >= x_bin_boundaries[i]) &
           (data["SLN"] < x_bin_boundaries[i+1])
        ].mean()

  cmp = sns.color_palette("deep")

  sns.scatterplot(
     data=data,
     x="SLN",
     y=xlabel,
     alpha=0.5
  )

  def regression(x, y):
     import statsmodels.api as sm
     from scipy.stats import zscore

     x2 = np.power(x, 2)
     x2 = zscore(x2)
     x2 = sm.add_constant(x2)

     model = sm.OLS(zscore(y), x2).fit()

     print(model.summary())

  regression(data["SLN"], data[xlabel])

  plt.plot(x_bin_center, y_average_center, color=cmp[1])
  plt.scatter(x_bin_center, y_average_center, color=cmp[1])
  # plt.fill_between(x_bin_center, y_average_center - y_std_center, y_average_center + y_std_center, color=cmp[1], alpha=0.5)
  # plt.ylim([-3, 10])

  ax=plt.gca()
  ax.set_xlabel(""+r"$SLN\left( i,j \right)$")
  # ax.set_ylabel(r"$D\left( i,j \right)$")
  ax.set_ylabel(r"$\log_{10}$["+"FLNe"+r"$\left( i,j \right)$]")
  # ax.set_ylabel(r"$D_{1/2}^{-}\left(i,j \right)$")

  sns.despine(offset=10, trim=True)

  plt.gcf().tight_layout()

  # plt.savefig(
  #    f"{H.plot_path}/sln/scatter_plot_sln_d{__nodes__}.svg",
  #   # f"{H.plot_path}/sln/scatter_plot_sln_flne_d{__nodes__}.svg",
  #   # f"{H.plot_path}/sln/scatter_plot_sln_D_d{__nodes__}.svg",
  #    transparent=True
  # )