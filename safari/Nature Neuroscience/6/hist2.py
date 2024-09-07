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
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
# plt.style.use("dark_background")
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

def histogram2(x, y, bins):
   xmin = np.min(x)
   xmax = np.max(x)
   xrange1 = np.linspace(xmin, xmax + 1e-3, bins+1)

   data = pd.DataFrame(
      {
         "X" : x,
         "Y" : y
      }
   )

   bin_height = np.zeros(bins)
   width = xrange1[1] - xrange1[0]

   for i in np.arange(bins):
      mybini = data["Y"].loc[(data["X"] >= xrange1[i]) & (data["X"] < xrange1[i+1])]
      bin_height[i] = mybini.sum() / (width * mybini.shape[0])

   xrange1[-1] -= 1e-3

   return bin_height

def  hist_plot(NET, pickle_path, ax : plt.Axes, cmap="deep"):


    path = pickle_path
    H = read_class(path, "hanalysis")

    SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

    A = NET.A[:NET.nodes, :][:, :NET.nodes]
    non_zero = A > 0  

    tgt = 1 -  H.target_sim_matrix
    # tgt[tgt != 0] = -2 * np.log(1 - tgt[tgt != 0])

    data = pd.DataFrame(
        {
            "D12" : tgt[non_zero],
            "SLN" : SLN[non_zero],
            "FLN" : -1/np.log10(A[non_zero])
        }
    )

    data["direction"] = "FF"
    data["direction"].loc[data["SLN"] < 0.5] = "FB"
    data["direction"].loc[data["SLN"] == 0.5] = "LAT"

    ax.tick_params(axis="y", labelleft=False)

    optheight = H.BH[0]["height"].loc[H.BH[0]["S"] == np.max(H.BH[0]["S"])].min()
    # optheight = -2 * np.log(1-optheight)

    ax.axhline(optheight, color="r", linestyle="--", linewidth=2, alpha=0.7)

    wff = data["FLN"].loc[(data["direction"] == "FF") & (data["D12"] <= optheight)]
    wfb = data["FLN"].loc[(data["direction"] == "FB") & (data["D12"] <= optheight)]

    from scipy.stats import ttest_ind

    print(ttest_ind(wff, wfb, equal_var=False).pvalue)

    sns.histplot(
       data=data,
       y="D12",
       weights="FLN",
       hue="direction",
      #  kde=T,
       multiple="dodge",
       bins=8,
      #  alpha=0.5,
       legend=F,
       ax=ax
    )

    ax.minorticks_on()
    ax.set_xlabel("Sum")
    # ax.set_ylim(bottom=-0.5, top=12)
    ax.tick_params(axis='y', which='both', left=False)