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
# plt.style.use("dark_background")
from pathlib import Path
# Personal libraries ----
from modules.hierarmerge import Hierarchy
from networks.structure import STR
from various.network_tools import *
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator


def optimality(H, host : plt.Axes, cmap="deep"):
    par = host.twinx()
    parx = host.twiny()
    par.minorticks_on()
    parx.minorticks_on()

    height = H.BH[0]["height"].to_numpy()

    k = H.BH[0]["K"].to_numpy()
    d = H.BH[0]["D"].to_numpy()
    s = H.BH[0]["S"].to_numpy()

    best_s_index = np.argmax(s)

    height2 = list(height[s < best_s_index][::10]) + list(height[s >= best_s_index])

    k2 = list(k[s < best_s_index][::10]) + list(k[s >= best_s_index])
    d2 = list(d[s < best_s_index][::10]) + list(d[s >= best_s_index])
    s2 = list(s[s < best_s_index][::10]) + list(s[s >= best_s_index])

    cmp = sns.color_palette(cmap)

    host.plot(height2, s2, color=cmp[0], linewidth=2, label="Loop Entropy", alpha=0.8)
    par.plot(height2, d2, color=cmp[1], linestyle="--", linewidth=2, label="Link community density", alpha=0.8)

    parx.plot(k2, s2, color=cmp[0], linewidth=7, label="Loop Entropy", alpha=0)

    host.plot(height[best_s_index], s[best_s_index], "*", markersize=10, color=cmp[3], markeredgecolor="k")

    print("Best index S:\t", height[best_s_index])


    host.set_xlabel(r"$H^{2}$")
    host.set_ylabel(r"$S_{L}$ (loop entropy)", fontsize=9)
    par.set_ylabel("Average LC density", fontsize=9)
    parx.set_xlabel("Number of LCs")

    import matplotlib.ticker as ticker

    host.xaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5]))
    par.xaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5]))
    parx.xaxis.set_major_locator(ticker.FixedLocator([1, 500, 999]))

    maxheight = np.max(height2)

    host.set_xlim(right=0.55)
    par.set_xlim(right=0.55)
    parx.set_xlim(left=1-((0.55-maxheight) / maxheight) * (999))

    parx.invert_xaxis()

    host.yaxis.label.set_color(cmp[0])
    par.yaxis.label.set_color(cmp[1])

    host.tick_params(axis="y", colors=cmp[0], which="both")
    par.tick_params(axis="y", colors=cmp[1], which="both")

    # lines = p1 + p2

    # host.legend(lines, [l.get_label() for l in lines], fontsize=12)

    # plt.show()
