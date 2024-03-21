# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Any
import os
# Personal libs ----
from various.network_tools import *
from modules.flatmap import FLATMAP

class Plot_NAKED:
  def __init__(self, A, D, labels, plot_path) -> None:
    ## Attributes ----
    
    self.A = A
    self.D = D
    self.labels = labels
    self.plot_path = plot_path
  
  def plot_consensus_heatmap(self, on=True):
    if on:
      _, ax = plt.subplots(1, 1, figsize=(10, 8))
      sns.heatmap(
        self.D,
        xticklabels=self.labels,
        yticklabels=self.labels,
        ax=ax
      )

      ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 8)
      ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 8)
      # Arrange path ----
      plot_path = join(self.plot_path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "consensus_distance_heatmap.png"
        ),
        dpi=300
      )
      plt.close()
  
  def core_dendrogram(self, Z, R : list, cmap_name="hls", leaf_font_size=20, remove_labels=False, on=True, **kwargs):
    if on:
      from scipy.cluster import hierarchy
      import matplotlib.colors
      # Create figure ----
      for r in R:
        if r == 1:
          r += 1
        partition = hierarchy.cut_tree(Z, r).ravel()
        new_partition = skim_partition(partition)
        unique_clusters_id = np.unique(new_partition)
        cm = sns.color_palette(cmap_name, len(unique_clusters_id))
        # dlf_col = "#808080"
        dlf_col = "#808080"
        ##
        D_leaf_colors = {}
        for i, _ in enumerate(self.labels):
          if new_partition[i] != -1:
            D_leaf_colors[i] = matplotlib.colors.to_hex(cm[new_partition[i]])
          else: D_leaf_colors[i] = dlf_col
        ##
        link_cols = {}
        for i, i12 in enumerate(Z[:,:2].astype(int)):
          c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x]
            for x in i12)
          link_cols[i+1+len(Z)] = c1 if c1 == c2 else dlf_col
        plt.style.use("dark_background")
        sns.set_context("talk")
        fig, ax = plt.subplots(1, 1)
        ax.grid(False)
        if not remove_labels:
          hierarchy.dendrogram(Z,
            labels=self.labels,
            color_threshold=Z[len(self.labels) - r, 2],
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, leaf_font_size=leaf_font_size, **kwargs
          )
        else:
          hierarchy.dendrogram(Z,
            no_labels=True,
            color_threshold=Z[len(self.labels) - r, 2],
            link_color_func = lambda k: link_cols[k]
          )
        fig.set_figwidth(13)
        fig.set_figheight(7)
        plt.ylabel("Height " + r"$(H^{2})$")
        sns.despine()
        # Arrange path ----
        plot_path = join(self.plot_path, "Features")
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          join(
            plot_path, f"core_dendrogram_{R}.png"
          ),
          dpi=300
        )
        plt.close()
  
  def plot_network_covers(self, R : npt.NDArray, partition : list, nocs : dict, sizes : dict, ang=0, score="", direction="", cmap_name="hls", figsize=(12,12), spring=False, on=True, **kwargs):
    if on:
      print("Printing network space")
      plt.style.use("dark_background")
      import matplotlib.patheffects as path_effects
      # Skim partition ----
      unique_clusters_id = np.unique(partition)
      keff = len(unique_clusters_id)
      # Generate all the colors in the color map -----
      if -1 in unique_clusters_id:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        save_colors = sns.color_palette(cmap_name, keff)
        cmap_heatmap = [[]] * (keff+1)
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {
          k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
        }
      else:
        nodes_memberships = {
          k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
        }
      for i, id in enumerate(partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(self.labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(len(self.labels)):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
      # Generate graph ----
      G = nx.from_numpy_array(R, create_using=nx.DiGraph)
      if "coords" not in kwargs.keys():
        pos = nx.kamada_kawai_layout(G, weight="weight")
        if spring:
          Ginv = nx.DiGraph(R)
          pos = nx.spring_layout(Ginv, weight="weight", pos=pos, iterations=5, seed=212)
      else:
        pos = kwargs["coords"]
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      labs = {k : lab for k, lab in zip(G.nodes, self.labels)}

      mu_pos_x = np.mean([k[0] for k in pos.values()])
      mu_pos_y = np.mean([k[1] for k in pos.values()])
      mu_pos = np.array([mu_pos_x, mu_pos_y])

      pos = {k : pos[k] - mu_pos for k in pos.keys()}
      pos = {k : pos[k] * 1.5 for k in pos.keys()}
      
      _, ax = plt.subplots(1, 1, figsize=figsize)
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(
          G, pos=pos, edge_color="#666666", alpha=0.5, width=2, arrowsize=10, connectionstyle="arc3,rad=-0.1",
          node_size=1400, ax=ax
        )
      if "modified_labels" not in kwargs.keys():
        t = nx.draw_networkx_labels(G, pos=pos, labels=labs, font_color="white", ax=ax)
        for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
          ]
        )
      else:
        t = nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"], font_color="white", ax=ax)
        for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
          ]
        )

      for node in G.nodes:
        a = plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node],  
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.08
        )
        for i in range(len(a[0])):
          a[0][i].set_alpha(0.8)
      array_pos = np.array([list(pos[v]) for v in pos.keys()])
      plt.xlim(-0.05 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.05)
      plt.ylim(-0.05 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.05)
      # Arrange path ----
      plot_path = join(self.plot_path, "Network")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, f"{direction}{score}.png"
        ),
        dpi=300
      )
      plt.close()

  def plot_newick(self, newick : str, colregion, width=8, height=7, fontsize=10, rotation=0, branches_color="#666666", on=True):
    if on:
      from Bio import Phylo
      from io import StringIO
      from matplotlib.transforms import Affine2D
      import mpl_toolkits.axisartist.floating_axes as floating_axes

      tree = Phylo.read(StringIO(newick), "newick")
      tree.ladderize()
      
      plt.style.use("dark_background")
      sns.set_context("talk")
      fig, ax = plt.subplots(1, 1, figsize=(width, height))

      # # set the figure size
      # plt.rcParams["figure.figsize"] = [width, height]
      # plt.rcParams["figure.autolayout"] = True
      # scales = (0, 1, 0, len(self.labels))

      # # Add 2D affine transformation
      # t = Affine2D().rotate_deg(90)

      # # plot the figure
      # fig = plt.figure()
      # # Add floating axes
      # h = floating_axes.GridHelperCurveLinear(t, scales)
      # ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=h)

      # fig.add_subplot(ax)
      
      ax.grid(False)
      area = colregion.regions.AREA.to_numpy().astype(str)
      color = colregion.regions.COLOR.to_numpy()
      color_tip = {k: v for k, v in zip(area, color)}
      Phylo.draw(
        tree, axes=ax, label_colors=color_tip,
        do_show=False, fontsize=fontsize, branches_color=branches_color,
        rotation=rotation
      )
      plt.xticks(rotation=180)
      ax.set_ylabel("")
      ax.set_xlabel(r"Height $H^{2}$", rotation=180)
      ax.yaxis.set_tick_params(labelleft=False)
      # ax.transAxes.rotate(90)
      sns.despine()
      # Arrange path ----
      plot_path = join(self.plot_path, "Newick")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "tree_H.png"
        ),
        dpi=300
      )
      plt.close()

  
  