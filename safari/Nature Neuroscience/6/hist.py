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

def  hist_plot(NET, pickle_path, ax : plt.Axes, cmap="deep"):


    path = pickle_path
    H = read_class(path, "hanalysis")

    SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

    A = NET.A[:NET.nodes, :][:, :NET.nodes]
    non_zero = A > 0  

    # label = r"$D_{1/2}^{\pm}$"

    tgt = 1 -  H.target_sim_matrix
    tgt[tgt != 0] = -2 * np.log(1 - tgt[tgt != 0])

    data = pd.DataFrame(
        {
            "D12" : tgt[non_zero],
            "SLN" : SLN[non_zero]
        }
    )

    data["direction"] = "FF"
    data["direction"].loc[data["SLN"] < 0.5] = "FB"
    data["direction"].loc[data["SLN"] == 0.5] = "LAT"

    ax.tick_params(axis="y", labelleft=False)

    optheight = H.BH[0]["height"].loc[H.BH[0]["S"] == np.max(H.BH[0]["S"])].to_numpy()
    optheight = -2 * np.log(1-optheight)

    ax.axhline(optheight, color="r", linestyle="--", linewidth=2, alpha=0.7)
 
    sns.histplot(
        data=data,
        y="D12",
        hue="direction",
        hue_order=["FF", "FB", "LAT"],
        stat="density",
        # common_norm=False,
        multiple="stack",
        bins=20,
        ax=ax,
        legend=False
    )

    ax.minorticks_on()
    ax.set_ylim(bottom=-0.5, top=12)
    ax.tick_params(axis='y', which='both', left=False)
    # ax.set_xlabel("Density")
