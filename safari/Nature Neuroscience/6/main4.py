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
from smile2 import smile2_plot
from line import line_plot
from hist import hist_plot

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

    fig, ax = plt.subplots(1, figsize=(3.54331, 3))
  
    ax.minorticks_on()

    line_plot(NET, H, ax) 

    sns.despine(ax=ax, top=True, right=True)

    
    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/6/Figure6_14.svg", bbox_inches='tight'
    # )
    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/6/Figure6_14.pdf", bbox_inches='tight'
    # )