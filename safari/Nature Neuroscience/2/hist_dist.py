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

def  hist_plot(nodes, pickle_path, ax : plt.Axes, cmap="deep"):


    path = pickle_path
    H = read_class(path, "hanalysis")

    label = r"$H^{2}_{\pm}$"
    label_inset = "Interareal distances [mm]"

    src = (1 - H.source_sim_matrix)
    tgt = (1 - H.target_sim_matrix)
    D = H.D[:nodes, :nodes]

    src = adj2df(src)
    tgt = adj2df(tgt)
    D = adj2df(D)

    src = src["weight"].loc[src.source > src.target].to_numpy()
    tgt = tgt["weight"].loc[tgt.source > tgt.target].to_numpy()
    D = D["weight"].loc[D.source > D.target].to_numpy()

    ordD = np.argsort(D)
    D = D[ordD]
    src = src[ordD]
    tgt = tgt[ordD]

    H_thr = 1-np.exp(-1)
    
    n = src.shape[0]

    special_areas_src = (np.sum(src < H_thr) * 100 / n)
    special_areas_tgt = (np.sum(tgt < H_thr) * 100 / n) 

    print("Area pairs src", special_areas_src)
    print("Area pairs tgt", special_areas_tgt)

    data_main = {
        label : np.hstack([src, tgt]),
        "dataset" : ["+ (out)"] * n + ["- (in)"] * n
    }

    cmp = sns.color_palette(cmap)

    sns.histplot(
        data=data_main,
        x=label,
        palette=cmp[:2],
        hue="dataset",
        hue_order=["+ (out)", "- (in)"],
        stat="density",
        common_norm=False,
        alpha=1,
        multiple="dodge",
        ax=ax,
        legend=False
    )
    

    # axinset = ax.inset_axes([0.12, 0.5, 0.45, 0.4], transform=ax.transAxes)
    # axinset.minorticks_on()

    # sns.histplot(
    #     x=D,
    #     color=cmp[2],
    #     stat="density",
    #     alpha=1,
    #     ax=axinset
    # )

    # axinset.set_xlabel(label_inset, fontsize=10)
    # axinset.set_ylabel("")

    # ax.axvline(H_thr, 0, 1, linewidth=4, color="r", linestyle="--")
    # ax.text(
    #     0.35, 0.2, "(" + f"{special_areas_src:.0f}%, {special_areas_tgt:.0f}%" + ")", transform=ax.transAxes,
    #     ha="center", va="center"
    # )
    
    ax.set_xlabel(label)
    # sns.move_legend(ax, "center left", bbox_to_anchor =(H_thr, 0.65), ncol=1, title=None, frameon=True)    