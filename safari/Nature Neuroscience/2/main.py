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
from dist_funct import *
from hist_dist import *
from H2_funct import *
from D12_funct import *
from D12_hist import *

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
    cortex_letter_path = os.path.join(
       "../plots", subject, version,
       structure, nature, distance, f"{__inj__}", NET.analysis
    )
    conf = {
       "subject" : NET.subject,
       "structure" : NET.structure,
       "version" : NET.version,
       "distance" : NET.distance,
       "subfolder" : NET.analysis,
       "model_distbase" : model_distbase,
       "model_swaps" : model_swaps
    }

    sns.set_style("ticks")
    color_palette = "deep"

    fig = plt.figure(1, figsize=(7.08661, 7))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=2, hspace=1)

    ax = fig.add_subplot(gs[0:2, 0:4])
    ax.minorticks_on()
    H2_bin_plot(NET.nodes, pickle_path, ax, cmap=color_palette)
    ax.text(
      0.0 , 1.0, "a", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    ax = fig.add_subplot(gs[0:2, 4:8])
    ax.minorticks_on()
    ax.text(
      0.0, 1.0, "b", transform=ax.transAxes,  
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )
    D12_bin_plot(NET.nodes, pickle_path, ax, cmap=color_palette)
    
    sns.despine(top=True, bottom=False, left=False, right=True)

    ax = fig.add_subplot(gs[2:4, 0:4])
    ax.minorticks_on()
    ax.text(
      0.0, 1.0, "c", transform=ax.transAxes,  
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    D12_hist_plot(NET.nodes, pickle_path, ax, cmap=color_palette)
    
    sns.despine(top=True, bottom=False, left=False, right=True)


    ax = fig.add_subplot(gs[2:4, 4:8])
    ax.minorticks_on()
    ax.text(
      0.0, 1.0, "d", transform=ax.transAxes,  
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    hist_plot(NET.nodes, pickle_path, ax, cmap=color_palette)
    
    sns.despine(top=True, bottom=False, left=False, right=True)

    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/2/Figure2_7.svg", bbox_inches='tight'
    # )
    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/2/Figure2_7.pdf", bbox_inches='tight'
    # )