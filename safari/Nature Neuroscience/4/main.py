import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from matplotlib import gridspec, colors
import matplotlib as mpl

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
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

# Load plots ----
from optimality import optimality
from network_cover import network_cover_plot
from prefrontal_communities import prefrontal_network_plot
from network_no_covers import network_no_cover_plot
from flatmap import flatmap_91_plot

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 12


linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
nature = "original"
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
__nodes__ = 40
__inj__ = 40
distance = "MAP3D"
version = f"{__nodes__}" + "d" + "91"
model_distbase = "M"
model_swaps = "TWOMX_FULL" # 1K_DENSE

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

    pickle_path = NET.pickle_path
    color_palette = "deep"

    sns.set_style("ticks")

    H = read_class(pickle_path, "hanalysis")

    fig = plt.figure(1, figsize=(7.08661, 7))
    gs = gridspec.GridSpec(5, 4)
    gs.update(wspace=1, hspace=0.2)

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.minorticks_on()
    ax.text(
      0.0, 1.0, "a", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )
    optimality(H, ax, cmap=color_palette)

    ax = fig.add_subplot(gs[0:4, 2:4])
    ax.text(
      0.0, 1.0, "b", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    network_cover_plot(NET, H, ax, scale=1, cmap=color_palette)

    ax = fig.add_subplot(gs[2:5, 0:2])
    ax.text(
      0.0, 1.0, "c", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    # network_cover_plot(NET, H, ax, scale=1, cmap=color_palette, spring=F)
    prefrontal_network_plot(NET, H, ax, scale=1, cmap="hls", spring=F)

    ax = fig.add_subplot(gs[2:5, 2:4])
    ax.text(
      0.0, 1.0, "d", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    flatmap_91_plot(NET, H, ax)

    plt.savefig(
      "../Publication/Nature Neuroscience/Figures/4/Figure4_yellow_4.svg", bbox_inches='tight'
    )

    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/4/Figure4_yellow.pdf", bbox_inches='tight'
    # )