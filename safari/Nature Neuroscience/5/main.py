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
from omega import omega
from overlap import overlap
from D12_null import three_divergences

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
fitter = "EXPTRUNC"
bins = 12
model_swaps = "1k" # 1K_DENSE

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
       "fitter" : fitter,
       "bins" : bins,
       "model_swaps" : model_swaps
    }

    H = read_class(NET.pickle_path, "hanalysis")

    sns.set_style("ticks")

    H = read_class(pickle_path, "hanalysis")

    fig = plt.figure(1, figsize=(7.08661, 2.5))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.35, hspace=0)

    ax = fig.add_subplot(gs[0:1, 0:1])
    ax.minorticks_on()
    omega(conf, ax, mode=mode, iterations=1000)
    ax.text(
      0.0, 1.0, "a", transform=ax.transAxes,
      fontsize=20, va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    sns.despine(ax=ax, top=True, right=True)

    ax = fig.add_subplot(gs[0:1, 1:2])
    ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    ax.text(
      0.0, 1.0, "b", fontsize=20, transform=ax.transAxes,
       va='top', fontfamily='sans-serif', ha="right", weight="bold"
    )

    three_divergences(ax, mode=mode, iterations=10)

    sns.despine(ax=ax, top=True, right=True)

    plt.savefig(
      "../Publication/Nature Neuroscience/Figures/5/Figure5_9_M_BIN12_1000_2.svg", bbox_inches='tight'
    )
    plt.savefig(
      "../Publication/Nature Neuroscience/Figures/5/Figure5_9_M_BIN12_1000_2.pdf", bbox_inches='tight'
    )

