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
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

# Load plots ----
from network_distsim import network_cover_plot

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 12
sns.set_style("ticks")


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
index  = "dist_sim"
bias = float(0)
alpha = 0.
__nodes__ = 40
__inj__ = 40
distance = "MAP3D"
version = f"{__nodes__}" + "d" + "91"
model_distbase = "M"
model_swaps = "TWOMX_FULL"

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

    H = read_class(pickle_path, "hanalysis")
    color_palette = "deep"

    fig = plt.figure(1, figsize=(5.35433, 8))
    gs = gridspec.GridSpec(6, 1)
    gs.update(wspace=0, hspace=0.5)

    ax = fig.add_subplot(gs[0:3, 0:1])
    ax.minorticks_on()
    ax.text(
      0.0, 1.0, "a", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )
    network_cover_plot(NET, H, ax, cmap=color_palette)
    

    ax = fig.add_subplot(gs[3:6, 0:1])
    # ax.minorticks_on()
    ax.text(
      0.0, 1.0, "b", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )
  
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)

    
    plt.savefig(
      "../Publication/Nature Neuroscience/ExtendedFigures/DistSimAnalysis/DistSim.svg", bbox_inches='tight'
    )
    plt.savefig(
      "../Publication/Nature Neuroscience/ExtendedFigures/DistSimAnalysis/DistSim.pdf", bbox_inches='tight'
    ) 