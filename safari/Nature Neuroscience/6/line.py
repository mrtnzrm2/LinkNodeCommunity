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

def line_plot(NET, H, ax : plt.Axes):

  D12 = 1 - H.target_sim_matrix
  # D12[D12 != 0] = -2 * np.log(D12[D12 != 0])

  SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

  A = NET.A[:NET.nodes, :][:, :NET.nodes]
  non_zero = A > 0  

  flnlabel = r"$\log_{10}$" + " FLN"

  data = pd.DataFrame(
     {
        # "D" : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        flnlabel : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        "D12" : D12[non_zero]
     }
  )

  data["direction"] = "FF"
  data["direction"].loc[data["SLN"] < 0.5] = "FB"
  data["direction"].loc[data["SLN"] == 0.5] = "LAT"
  data["SLN2"] = np.abs(data["SLN"]-0.5)

  true_bins = 8
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

  from scipy.stats import ttest_ind

  for i in np.arange(true_bins):
     
     dyff = data[xlabel].loc[
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1]) &
           (data["direction"] == "FF")
     ]
     dyfb = data[xlabel].loc[
           (data["SLN2"] >= x_bin_boundaries[i]) &
           (data["SLN2"] < x_bin_boundaries[i+1]) &
           (data["direction"] == "FB")
      ]
     
     print(ttest_ind(dyff, dyfb, equal_var=False).pvalue)

     yff_average_center[i] = dyff.mean()
     yff_std_center[i] = dyff.std()
     
     yfb_average_center[i] = dyfb.mean()
     yfb_std_center[i] = dyfb.std()
     


  cmp = sns.color_palette("deep")

  sns.scatterplot(
     data=data,
     x="SLN2",
     size=flnlabel,
     y=xlabel,
     hue="direction",
     style="direction",
     markers=["o", "^", "s"],
    #  s=20 * 3/6,
    sizes=(5, 20),
     alpha=0.5,
     legend=T,
     ax=ax
  )

  from scipy.stats import pearsonr, spearmanr


  dataFF = [data[xlabel].loc[data["direction"] == "FF"], data["SLN2"].loc[data["direction"] == "FF"]]
  dataFB = [data[xlabel].loc[data["direction"] == "FB"], data["SLN2"].loc[data["direction"] == "FB"]]

  print(pearsonr(dataFF[0], dataFF[1]))
  print(pearsonr(dataFB[0], dataFB[1]))

  print(spearmanr(dataFF[0], dataFF[1]))
  print(spearmanr(dataFB[0], dataFB[1]))

  print(dataFF[0].shape)
  print(dataFB[0].shape)

  
  ax.plot(x_bin_center, yff_average_center, color=cmp[0])
  
  ax.fill_between(
    x_bin_center,
    yff_average_center + yff_std_center / 2,
    yff_average_center - yff_std_center / 2, color=cmp[0],
    alpha=0.4, edgecolor=None
  )

  # ax.errorbar(
  #    x_bin_center,
  #    yff_average_center,
  #    1.96 * yff_std_center/2,
  #    color=cmp[0],
  #    alpha=0.6, capsize=5
  # )

  ax.scatter(x_bin_center, yff_average_center, color=cmp[0])
  # ax.scatter(x_bin_center, yff_average_center, color=cmp[0])

  
  ax.plot(x_bin_center, yfb_average_center, color=cmp[1])
  
  ax.fill_between(
    x_bin_center,
    yfb_average_center + yfb_std_center / 2,
    yfb_average_center - yfb_std_center / 2, color=cmp[1],
    alpha=0.4, edgecolor=None
  )

  # ax.errorbar(
  #    x_bin_center,
  #    yfb_average_center,
  #    1.96 * yfb_std_center/2,
  #    color=cmp[1],
  #    alpha=0.6, capsize=5
  # )

  ax.scatter(x_bin_center, yfb_average_center, color=cmp[1])

  # ax.scatter(x_bin_center, yfb_average_center, color=cmp[1])

  optheight = H.BH[0]["height"].loc[H.BH[0]["S"] == np.max(H.BH[0]["S"])].min()

  print(H.BH[0].loc[H.BH[0]["S"] == np.max(H.BH[0]["S"])])
  print(H.BH[0].shape)
  

  print(">>>", optheight)
  ax.axhline(optheight, color="r", linestyle="--", linewidth=2, alpha=0.7)

  ###

  nff = data.loc[(data["D12"] <= optheight) & (data["direction"] == "FF")].shape[0]
  nfb = data.loc[(data["D12"] <= optheight) & (data["direction"] == "FB")].shape[0]
  print(">>>nt", (nff-nfb)/nfb, nff, nfb)

  tnff = data.loc[data["direction"] == "FF"].shape[0]
  tnfb = data.loc[data["direction"] == "FB"].shape[0]

#   print(tnff/tnfb)
  print(">>>t", (tnfb - tnff) / tnff, tnff, tnfb)


#   aflnFF = data[flnlabel].loc[(data["D12"] <= optheight[0]) & (data["direction"] == "FF")].mean()
#   aflnFB = data[flnlabel].loc[(data["D12"] <= optheight[0]) & (data["direction"] == "FB")].mean()

#   print(np.abs((aflnFF - aflnFB) / aflnFB))

  ###

  ax.set_xlabel(r"$|SLN-1/2|$")
  # ax.set_ylabel(r"$D_{1/2}^{-}$")
  ax.set_ylabel(r"$H^{2}_{-}$")

  leg = ax.get_legend()
  leg.set_bbox_to_anchor([1.5, 1], transform=ax.transAxes)