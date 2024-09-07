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
from networks.swapnet import SWAPNET
from networks.structure import STR
from various.network_tools import *
import matplotlib as mpl
from various.data_transformations import maps

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 20
sns.set_style("ticks")

def  link_community_h_conf_plot(ax : plt.Axes, cmap="deep"):
    from scipy.cluster.hierarchy import dendrogram

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

    RAND = SWAPNET(
      __inj__,
      91,
      linkage,
      mode, 0,
      structure = structure,
      version = "40d91",
      topology=topology,
      nature = nature,
      distance = distance,
      model = "1k",
      mapping=mapping,
      index=index,
      nlog10 = nlog10, lookup = lookup,
      cut=cut, b=bias, discovery=discovery,
      subject=subject
    )

    RAND.C, RAND.A = NET.C, NET.A
    RAND.D = NET.D

    # Create network ----
    print("Create random graph")
    RAND.random_one_k(run=T, swaps=1000000, on_save_csv=F)

    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RAND.A, nlog10, lookup, prob, b=bias
    )

    # Compute RAND Hierarchy ----
    print("Compute Hierarchy")
    H = Hierarchy(
      RAND, RAND.A[:, :__nodes__], R[:, :__nodes__], RAND.D,
      __nodes__, linkage, mode, lookup=lookup, alpha=alpha,
      index=index
    )

    

    h = H.H.copy()
    # h[:, 2] = h[:, 2] / np.max(NET.D)
    # h[:, 2] = np.sqrt(h[:, 2])

    cmp = sns.color_palette(cmap)

    dendrogram(
        h,
        no_labels=True,
        orientation= "top",
        color_threshold=0,
        above_threshold_color=mpl.colors.rgb2hex((76/255, 114/255, 176/255, 0.7), keep_alpha=True),
        ax=ax
    )

    ax.set_ylabel(r"$H^{2}$")
    # ax.set_title("Configuration model (same)")
    # plt.show()
    
    # sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5), ncol=1, title=None, frameon=False)
    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/2/dis_funct_v1.pdf"
    # )
    # plt.show()

    