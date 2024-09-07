# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
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

def smile_plot(NET, H, ax : plt.Axes):

  D12 = H.target_sim_matrix
  D12[D12 != 0] = -2 * np.log(D12[D12 != 0])

  D12p = H.source_sim_matrix
  D12p[D12p != 0] = -2 * np.log(D12p[D12p != 0])

  SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

  A = NET.A[:NET.nodes, :][:, :NET.nodes]
  non_zero = A > 0

  data = pd.DataFrame(
     {
        "D" : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        "FLNe" : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        "D12" : D12[non_zero]
     }
  )


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
  y_average_center = np.zeros(x_bin_center.shape[0])
  y_std_center = np.zeros(x_bin_center.shape[0])

  xlabel = "D12"

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
     s=6,
     alpha=0.5,
     ax=ax
  )

  ax.plot(x_bin_center, y_average_center, color=cmp[1])
  ax.scatter(x_bin_center, y_average_center, color=cmp[1])
  ax.set_xlabel(""+r"$SLN\left( i,j \right)$")
  ax.set_ylabel(r"$D_{1/2}^{-}\left( i,j \right)$")
  # ax.set_ylabel(r"$D\left( i,j \right)$")


  flne_label = r"$\log_{10} FLNe$"
  d12_label = r"$D_{1/2}^{-}$"
  dist_label = "MAP3D"

  data = pd.DataFrame( 
     {
        dist_label : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        flne_label : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        d12_label : D12[non_zero]
        # "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
     }
  )

  def regression(x, y):
     import statsmodels.api as sm
     from scipy.stats import zscore

     x2 = np.power(x, 2)
     x2 = zscore(x2)
     x2 = sm.add_constant(x2)

     model = sm.OLS(zscore(y), x2).fit()
     print(model.summary())
     return model.fvalue, model.f_pvalue
  
  data_ftest = pd.DataFrame()

  for ylabel in [d12_label, flne_label, dist_label]:
    f, fp = regression(data["SLN"], data[ylabel])
    data_ftest = pd.concat(
      [
        data_ftest,
        pd.DataFrame(
          {
            "Dependent variable" : [ylabel],
            "F-statistic" : [f],
            "P-value" : [fp]
          }
        )
      ], ignore_index=True
    )

  # print(data_ftest)

  # axinset = ax.inset_axes([0.5, 0.55, 0.35, 0.3], transform=ax.transAxes)
  # axinset.minorticks_on()
  

  # sns.scatterplot(
  #   data=data_ftest,
  #   x="P-value",
  #   y="F-statistic",
  #   hue="Dependent variable",
  #   s=30,
  #   palette="hls",
  #   ax=axinset
  # )

  # data_ftest = data_ftest.sort_values("P-value")
  # axinset.plot(
  #   data_ftest["P-value"].to_numpy(),
  #   data_ftest["F-statistic"].to_numpy(),
  #   color="k"
  # )

  # axinset.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.4))