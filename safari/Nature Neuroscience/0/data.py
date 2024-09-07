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
from plotting_modules.plotting_H import Plot_H
from various.network_tools import *

from matplotlib import gridspec, colors
import matplotlib as mpl

def heatmap(H, r, R, score="", cmap="viridis", cbar_label="", suffix="", font_color = None, center=None, linewidth=1.5, on=True, **kwargs):
    if on:
      print("Plot heatmap structure!!!")
      # plt.box()
      # Transform FLNs ----
      W = R.copy()
      rows, cols = R.shape
      W[W == 0] = np.nan
      W[W == -np.Inf] = np.nan
      # Get nodes ordering ----
      # from scipy.cluster import hierarchy
      # den_order = np.array(
      #   hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
      # ).astype(int)
      # memberships = hierarchy.cut_tree(self.Z, r).ravel()
      # memberships = skim_partition(memberships)[den_order]
      # C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      # D = np.where(memberships == -1)[0] + 1
      # C = list(set(C).union(set(list(D))))
      # W = W[den_order, :][:, den_order]
      # Configure labels ----
      labels = H.colregion.labels
      labels =  np.char.lower(labels.astype(str))
      # colors = H.colregion.regions.loc[
      #   match(
      #     labels,
      #     rlabels
      #   ),
      #   "COLOR"
      # ].to_numpy()

      #permutation
      perm_col = np.random.permutation(cols)
      perm_row = np.random.permutation(np.arange(cols, rows))
      perm_row = np.hstack([perm_col, perm_row])
      W = W[perm_row, :][:, perm_col]
      labels = labels[perm_row]
      # colors = colors[perm]

      sns.set_style("ticks")
      sns.set_context("talk")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(10)
      fig.set_figheight(15)

      import matplotlib
      cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#8E1B16","#FDF399"])

      g=sns.heatmap(
        W,
        cmap=cmap,
        center=center,
        # linewidth=0.5,
        ax = ax,
        cbar_kws={"label" : cbar_label}
      )
      
      g.set_facecolor("#030103")
      
      # for c in C:
      #   ax.vlines(
      #     c, ymin=0, ymax=self.nodes,
      #     linewidth=linewidth,
      #     colors=["#f4ff22"]
      #   )
      #   ax.hlines(
      #     c, xmin=0, xmax=self.nodes,
      #     linewidth=linewidth,
      #     colors=["#f4ff22"]
      #   )
      
      import matplotlib.ticker as ticker

      ax.xaxis.set_ticklabels([])

      ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.5, 38.5, 20)))
      ax.xaxis.set_ticklabels([l for i, l in enumerate(labels[:cols]) if i % 2 == 0])
      # colors1 = [c for i, c in enumerate(colors) if i % 2 == 0]
      # [t.set_color(c) for c, t in zip(colors1, ax.xaxis.get_ticklabels())]

      ax2 = ax.twiny()
      ax2.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5, 39.5, 20)/40))
      ax2.xaxis.set_ticklabels([l for i, l in enumerate(labels[:cols]) if i % 2 == 1])
      # colors2 = [c for i, c in enumerate(colors) if i % 2 == 1]
      # [t.set_color(c) for c, t in zip(colors2, ax2.xaxis.get_ticklabels())]

      ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.5, 51 + 38.5, 46)))
      ax.yaxis.set_ticklabels([l for i, l in enumerate(labels) if i % 2 == 0])
      # [t.set_color(c) for c, t in zip(colors1, ax.yaxis.get_ticklabels())]

      ax3 = ax.twinx()
      ax3.yaxis.set_inverted(True)
      ax3.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5, 51 + 39.5, 45)/91))
      ax3.yaxis.set_ticklabels([l for i, l in enumerate(labels) if i % 2 == 1])
      # [t.set_color(c) for c, t in zip(colors2, ax3.yaxis.get_ticklabels())]
      
      # Setting labels colors ----
      # [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
      # [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]

      # if font_color:
      #   [t.set_color(font_color) for t in ax.xaxis.get_ticklabels()]
      #   [t.set_color(font_color) for t in ax.yaxis.get_ticklabels()]

      if "fontsize" in kwargs.keys():
        if kwargs["fontsize"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["fontsize"], rotation=90
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["fontsize"], rotation=0
          )
          ax2.set_xticklabels(
            ax2.get_xmajorticklabels(), fontsize = kwargs["fontsize"], rotation=90
          )
          ax3.set_yticklabels(
            ax3.get_ymajorticklabels(), fontsize = kwargs["fontsize"], rotation=0
          )

      ax.set_ylabel("Source", fontdict={"fontsize" : 30})
      ax.set_xlabel("Target", fontdict={"fontsize" : 30})
      # Arrange path ----
      # plot_path = os.path.join(
      #   self.path, "Heatmap_single"
      # )

      # plt.show()
      # Crate path ----
      # Path(
      #     plot_path + "/svg/"
      # ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plot_path = "../Publication/Nature Neuroscience/Figures/00/NFig1_3.svg"
      plt.savefig(plot_path)
    #   # Crate path ----
    #   Path(
    #       plot_path + "/png/"
    #   ).mkdir(exist_ok=True, parents=True)
    #   # Save plot ----
    #   plt.savefig(
    #     os.path.join(
    #       plot_path + "/png/", f"dendrogram_order_{r}{score}{suffix}.png"
    #     ),
    #     dpi = 300
    #   )
    #   plt.close()
    # else:
    #   print("No heatmap structure")

# Load plots ----

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
    H = read_class(pickle_path, "hanalysis")

    plot_h = Plot_H(NET, H)

    RW10 = NET.A.copy()
    RW10[RW10 > 0] = -np.log10(RW10[RW10 > 0])
    np.fill_diagonal(RW10, 0.)
    heatmap(H, 0, -RW10, on=T, linewidth=3, score="FLNe", cbar_label=r"$\log_{10}$FLN", fontsize = 25, suffix="2")



