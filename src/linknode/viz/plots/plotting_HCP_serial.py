# Standard libs ----
import os
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import networkx as nx
from pathlib import Path
from os.path import join
# Personal libs ----
from various.network_tools import *
from plotting_modules.plotting_o_serial import PLOT_OS

class PLOT_HCP(PLOT_OS):
  def __init__(self, hrh) -> None:
    super().__init__(hrh)

  def plot_newick_R_PIC(self, tree_newick, picture_path, on=True):
    if on:
      print("Plot tree in Newick format from R!!!")
      import subprocess
      # Arrange path ----
      plot_path = join(self.plot_path, "NEWICK")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      subprocess.run(["Rscript", "R/plot_newick_tree_PIC.R", tree_newick, join(plot_path, "tree_newick_pic.png"), picture_path])
    else:
      print("No tree in Newick format")
  
  def heatmap_dendro(self, r, R, Z, L, score="", cmap="viridis", center=None, linewidth=1.5, on=True, **kwargs):
    if on:
      print("Plot heatmap structure!!!")
      plt.box()
      # Transform FLNs ----
      W = R.copy()
      W[W == 0] = np.nan
      W[W == -np.Inf] = np.nan
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(
        hierarchy.dendrogram(Z, no_plot=True)["ivl"]
      ).astype(int)
      memberships = hierarchy.cut_tree(Z, r).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      W = W[den_order, :][:, den_order]
      # Configure labels ----
      labels = L.labels
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = [
        str(re) for re in L.regions[
          "AREA"
        ]
      ]
      colors = L.regions.loc[
        match(
          labels,
          rlabels
        ),
        "COLOR"
      ].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(18)
      fig.set_figheight(15)
      sns.heatmap(
        W,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        center=center,
        ax = ax
      )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=linewidth,
          colors=["#C70039"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=linewidth,
          colors=["#C70039"]
        )
      # Setting labels colors ----
      [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
      [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]
      plt.xticks(rotation=90)
      plt.yticks(rotation=0)
      plt.ylabel("Source")
      plt.xlabel("Target")
      # Arrange path ----
      plot_path = os.path.join(
        self.plot_path, "Heatmap_single"
      )
      # Crate path ----
      Path(
        plot_path
    ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"dendrogram_order_{r}{score}.png"
        ),
        dpi = 300
      )
      plt.close()
    else:
      print("No heatmap structure")

