# Standard libs ----
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import pandas as pd
# Personal libs ----
from various.network_tools import *
from plotting_modules.plotting_H import Plot_H

class PLOT_O(Plot_H):
  def __init__(self, Net, Hierarchy) -> None:
    super().__init__(Net, Hierarchy)

  def bar_node_membership(self, K, score="", on=True, **kwargs):
    if on:
      from scipy.cluster.hierarchy import cut_tree
      for k in K:
        print(
          "Plot bar node membership with K:\t{}!!!".format(k)
        )
        dA = self.dA.copy()
        dA["id"] = cut_tree(
          self.H,
          n_clusters=k
        ).reshape(-1)
        self.minus_one_Dc(dA)
        self.aesthetic_ids(dA)
        # Take out dc == 0 lcs ----
        dA = dA.loc[dA["id"] != -1]
        dA.id.loc[dA.id > 0] = dA.id.loc[dA.id > 0] - 1
        # unique ids ----
        Tids = np.unique(dA["id"])
        # Get labels and regions --- 
        labels = self.colregion.labels
        regions = self.colregion.regions
        I = np.arange(self.nodes).astype(int)
        if "labels" in kwargs.keys():
          I, _ = sort_by_size(kwargs["labels"], self.nodes)
        rlabels = [
          str(r) for r in self.colregion.regions[
            "AREA"
          ]
        ]
        colors = regions.COLOR.loc[ match(labels, rlabels)].to_numpy()
        # Create data ----
        data = bar_data(dA, self.nodes, labels)
        # Create figure ----
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(8)
        colors_ids = sns.color_palette("husl", len(Tids))
        colors_ids = list(colors_ids)
        bottom = np.zeros(self.nodes)
        for i, id in enumerate(Tids):
          x = data.nodes.loc[data.ids == id].to_numpy()
          sizes = np.zeros(self.nodes)
          if "node_labels" in kwargs.keys():
            x = [np.where(kwargs["node_labels"] == i)[0][0] for i in x]
          sizes[x] = data["size"].loc[data["ids"] == id].to_numpy()
          sizes = sizes[I]
          plt.bar(
            np.arange(self.nodes),
            sizes,
            color = colors_ids[i],
            bottom=bottom,
            label = id
          )
          bottom += sizes
        plt.xticks(np.arange(self.nodes), labels[I])
        # Setting labels colors ----
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.xaxis.get_ticklabels()
          )
        ]
        plt.xticks(rotation=90)
        # Arrange path ----
        plot_path = os.path.join(self.path, "Bar_ids")
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "k_{}{}.png".format(k, score)
          ),
          dpi = 300
        )

    else:
      print("No bar node membership")

  def bar_node_overlap(self, K, overlap, score="", on=True, **kwargs):
    if on:
      from scipy.cluster.hierarchy import cut_tree
      for k in K:
        print(
          "Plot bar node membership with K:\t{}!!!".format(k)
        )
        dA = self.dA.copy()
        dA["id"] = cut_tree(
          self.H,
          n_clusters=k
        ).reshape(-1)
        self.minus_one_Dc(dA)
        self.aesthetic_ids(dA)
        # Take out dc == 0 lcs ----
        dA = dA.loc[dA["id"] != -1]
        dA.id.loc[dA.id > 0] = dA.id.loc[dA.id > 0] - 1
        # unique ids ----
        Tids = np.unique(dA["id"])
        ### Special of this method ----
        o = overlap
        if "node_labels" in kwargs.keys():
          o = [np.where(kwargs["node_labels"] == i)[0][0] for i in o]
        # Get labels and regions --- 
        labels = self.colregion.labels
        rlabels = [
          str(r) for r in self.colregion.regions[
            "AREA"
          ]
        ]
        colors = self.colregion.regions.loc[
          match(
            labels,
            rlabels
          ),
          "COLOR"
        ].to_numpy()
        # Create data ----
        data = bar_data(
          dA, self.nodes, labels
        )
        # Create figure ----
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(8)
        colors_ids = sns.color_palette("husl", len(Tids))
        colors_ids = list(colors_ids)
        bottom = np.zeros(len(o))
        for i, id in enumerate(Tids):
          x = data.nodes.loc[data["ids"] == id].to_numpy()
          sizes = np.zeros(self.nodes)
          if "node_labels" in kwargs.keys():
            x = [np.where(kwargs["node_labels"] == i)[0][0] for i in x]
          sizes[x] = data["size"].loc[data["ids"] == id]
          plt.bar(
            np.arange(len(o)),
            sizes[o],
            color = colors_ids[i],
            bottom=bottom,
            label = id
          )
          bottom += sizes[o]
        plt.xticks(np.arange(len(o)), labels[o])
        # Setting labels colors ----
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.xaxis.get_ticklabels()
          )
        ]
        plt.xticks(rotation=90)
        # Arrange path ----
        plot_path = os.path.join(self.path, "Bar_ids_o")
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "k_{}{}.png".format(k, score)
          ),
          dpi = 300
        )

    else:
      print("No bar node membership")