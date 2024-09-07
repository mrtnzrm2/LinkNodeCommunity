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
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

from matplotlib import gridspec, colors
import matplotlib as mpl

# Load plots ----
from sln_matrix import sln_matrix_check_BB
from sln_scatter import sln_matrix_shuffle_test_BB
from smile import smile_plot
from smile2 import smile2_plot

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 8

# Declare iter variables ----
topology = "MIX"
bias = 0.
mode = "ZERO"
# Declare global variables NET ----
MAXI = 1000
linkage_criterion = "single"
nlog10 = F
lookup = F
prob = T
cut = F
run = T
subject = "MAC"
structure = "FLNe"
nature = "original"
mapping = "trivial"
index = "Hellinger2"
imputation_method = ""
opt_score = ["_S"]
sln = True
# Declare global variables ----
total_nodes = 91
__inj__ = 40
__nodes__ = 40
distance = "MAP3D"
__version__ = f"{__nodes__}d{total_nodes}"
__model__ = "TWOMX_FULL"
# T test ----
alternative = "less"
# Print summary ----
print("For NET parameters:")
print(
  "linkage: {}\nscore: {}\nnlog: {}\n lookup: {}".format(
    linkage_criterion, opt_score, nlog10, lookup
  )
)
print("For imputation parameters:")
print(
  "nature: {}\nmodel: {}".format(
    nature, imputation_method
  )
)
print("Random network and statistical paramteres:")
print(
  "nodes: {}\ninj: {}\nalternative: {}".format(
    str(__nodes__),str(__inj__), alternative
  )
)
# Start main ----
if __name__ == "__main__":
    bias = float(bias)
    l10 = ""
    lup = ""
    _cut = ""
    if nlog10: l10 = "_l10"
    if lookup: lup = "_lup"
    if cut: _cut = "_cut"
    print("Load MAC data ----")
    data = read_class(
      "../pickle/RAN/swaps/MAC/{}/{}/{}/{}/{}/{}/{}/{}".format(
        __version__,
        structure,
        distance,
        __model__,
       f"{linkage_criterion.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
        mode,
        f"{topology}_{index}_{mapping}",
        "discovery_7"
      ),
      "series_{}".format(MAXI)
    )

    if isinstance(data, int): raise

    NET = STR[f"{subject}{__inj__}"](
      "single", mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = "40d91",
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      discovery = "discovery_7",
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = 0.
    )
    pickle_path = NET.pickle_path

    sns.set_style("ticks")
    # sns.set_context("talk")

    H = read_class(pickle_path, "hanalysis")
    color_palette = "deep"

    fig = plt.figure(1, figsize=(7.20472, 3.5))
    gs = gridspec.GridSpec (2, 4)
    gs.update(wspace=0.5, hspace=0.25)

    ax = fig.add_subplot(gs[0:2, 0:2])
    # ax.minorticks_on()
   
    # sln_matrix_check_BB(data.cover_corr_sln, MAXI, ax)

    # ax.text(
    #   0.0, 1.0, "a", transform=ax.transAxes,
    #   fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    # )

    # # ax.tick_params(axis="x", which="minor", left=False, bottom=False)

    # ax = fig.add_subplot(gs[0:2, 2:4])
    # ax.minorticks_on()
    # # ax.tick_params(axis='x', which='minor', bottom=False)
    # ax.text(
    #   0.0, 1.0, "b", fontsize=20, transform=ax.transAxes,
    #    va='top', fontfamily='sans-serif', ha="right", weight="bold"
    # )

    # sln_matrix_shuffle_test_BB(data.cover_corr_sln, MAXI, ax)

    # sns.despine(ax=ax, top=True)

    # ax = fig.add_subplot(gs[2:4, 0:2])
    ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    ax.text(
      0.0, 1.0, "a", fontsize=20, transform=ax.transAxes,
       va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )
    
    smile2_plot(NET, H, ax)

    sns.despine(ax=ax, top=True, right=True)

    ax = fig.add_subplot(gs[0:1, 2:4])
    # ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    ax.text(
      0.0, 1.0, "b", fontsize=20, transform=ax.transAxes,
       va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine(ax=ax, top=True, left=True, bottom=True, right=True)

    ax = fig.add_subplot(gs[1:2, 2:4])
    # ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    ax.text(
      0.0, 1.0, "c", fontsize=20, transform=ax.transAxes,
       va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine(ax=ax, top=True, left=True, bottom=True, right=True)

    plt.savefig(
      "../Publication/Nature Neuroscience/Figures/6/Figure6_2.svg", bbox_inches='tight'
    )
    plt.savefig(
      "../Publication/Nature Neuroscience/Figures/6/Figure6_2.pdf", bbox_inches='tight'
    )