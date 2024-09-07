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

def smile2_plot(NET, H, ax : plt.Axes):

  D12 = 1 - H.target_sim_matrix
  D12[D12 != 0] = -2 * np.log(1 - D12[D12 != 0])

  SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

  A = NET.A[:NET.nodes, :][:, :NET.nodes]
  non_zero = A > 0  

  data = pd.DataFrame(
     {
        # "D" : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        # "FLNe" : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        "D12" : D12[non_zero]
     }
  )

  data["direction"] = "FF"
  data["direction"].loc[data["SLN"] < 0.5] = "FB"
  data["direction"].loc[data["SLN"] == 0.5] = "LAT"
  data["SLN2"] = np.abs(data["SLN"]-0.5)



  true_bins = 12
  bins = true_bins - 1
  min_x = np.min(data["SLN2"])
  max_x = np.max(data["SLN2"])
  x_bin_boundaries = np.linspace(min_x, max_x, bins+1)
  bin_width = x_bin_boundaries[1]-x_bin_boundaries[0]
  x_bin_center = x_bin_boundaries[1:] - bin_width
  x_bin_center = np.hstack([x_bin_center, np.array(x_bin_center[-1] + bin_width)])
  x_bin_boundaries -= bin_width / 2
  x_bin_boundaries = np.hstack([x_bin_boundaries, np.array(x_bin_boundaries[-1] + bin_width)])


  yff_average_center = np.zeros(x_bin_center.shape[0])
  yff_std_center = np.zeros(x_bin_center.shape[0])

  yfb_average_center = np.zeros(x_bin_center.shape[0])
  yfb_std_center = np.zeros(x_bin_center.shape[0])

  xlabel = "D12"

  for i in np.arange(true_bins):
     yff_average_center[i] = data[xlabel].loc[
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1]) &
           (data["direction"] == "FF")
        ].mean()
     yff_std_center[i] = data[xlabel].loc[
          #  (data["class"] == "-") &
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1])  &
           (data["direction"] == "FF")
        ].std()
     
     yfb_average_center[i] = data[xlabel].loc[
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1]) &
           (data["direction"] == "FB")
        ].mean()
     yfb_std_center[i] = data[xlabel].loc[
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1]) &
           (data["direction"] == "FB")
        ].std()

  cmp = sns.color_palette("deep")

  sns.scatterplot(
     data=data,
     x="SLN2",
     y=xlabel,
     hue="direction",
     style="direction",
     markers=["o", "^", "s"],
     s=20 * 3/6,
     alpha=0.5,
     ax=ax
  )

  ax.plot(x_bin_center, yff_average_center, color=cmp[0])
  ax.fill_between(
    x_bin_center,
    yff_average_center + yff_std_center / 2,
    yff_average_center - yff_std_center / 2, color=cmp[0],
    alpha=0.4, edgecolor=None
  )
  # ax.scatter(x_bin_center, yff_average_center, color=cmp[0])

  ax.plot(x_bin_center, yfb_average_center, color=cmp[1])
  ax.fill_between(
    x_bin_center,
    yfb_average_center + yfb_std_center / 2,
    yfb_average_center - yfb_std_center / 2, color=cmp[1],
    alpha=0.4, edgecolor=None
  )
  # ax.scatter(x_bin_center, yfb_average_center, color=cmp[1])

  optheight = H.BH[0]["height"].loc[H.BH[0]["S"] == np.max(H.BH[0]["S"])].to_numpy()
  optheight = -2 * np.log(1-optheight)
  ax.axhline(optheight, color="r", linestyle="--", linewidth=2, alpha=0.7)

  ###

  # nff = data.loc[(data["D12"] <= optheight[0]) & (data["direction"] == "FF")].shape[0]
  # nfb = data.loc[(data["D12"] <= optheight[0]) & (data["direction"] == "FB")].shape[0]
  # print((nff-nfb)/nfb)

  ###

  ax.set_xlabel(r"$|SLN-1/2|$")
  ax.set_ylabel(r"$D_{1/2}^{-}$")
  ax.set_ylim(bottom=-0.5, top=12)
  # ax.set_ylabel(r"$H^{2}_{-}$")


  import matplotlib.lines as mlines

  blue_line = mlines.Line2D( 
      [], [], color=cmp[0], label='FF', lw=2, marker="o", markersize=7.5, alpha=0.6, markeredgewidth=0.1
  )

  green_line = mlines.Line2D( 
      [], [], color=cmp[2], label='LAT', marker="s", markersize=7.5, alpha=0.6, markeredgewidth=0.1
  )
  orange_line = mlines.Line2D( 
      [], [], color=cmp[1], label='FB', lw=2, markersize=7.5, marker="^", alpha=0.6, markeredgewidth=0.1
  )

  legend = ax.legend(
    handles=[blue_line, green_line, orange_line],
    # bbox_to_anchor=(0.85, 0.2),
    loc="upper left",
    fontsize=8.5,
    ncol=3
  )
  
  # ax.set_ylabel(r"$D\left( i,j \right)$")


  # flne_label = r"$\log_{10} FLNe$"
  # d12_label = r"$D_{1/2}^{-}$"
  # dist_label = "MAP3D"

  # data = pd.DataFrame( 
  #    {
  #       dist_label : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
  #       flne_label : list(np.log10(A[non_zero])) * 1,
  #       "SLN" : list(SLN[non_zero]) * 1,
  #       d12_label : D12[non_zero]
  #       # "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
  #    }
  # )

  # def regression(x, y):
  #    import statsmodels.api as sm
  #    from scipy.stats import zscore

  #    x2 = np.power(x, 2)
  #    x2 = zscore(x2)
  #    x2 = sm.add_constant(x2)

  #    model = sm.OLS(zscore(y), x2).fit()
  #    print(model.summary())
  #    return model.fvalue, model.f_pvalue
  
  # data_ftest = pd.DataFrame()

  # for ylabel in [d12_label, flne_label, dist_label]:
  #   f, fp = regression(data["SLN"], data[ylabel])
  #   data_ftest = pd.concat(
  #     [
  #       data_ftest,
  #       pd.DataFrame(
  #         {
  #           "Dependent variable" : [ylabel],
  #           "F-statistic" : [f],
  #           "P-value" : [fp]
  #         }
  #       )
  #     ], ignore_index=True
  #   )

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