# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
import os
# Personal libs ----
from various.network_tools import *
from modules.flatmap import FLATMAP

class Plot_H:
  def __init__(self, NET, H) -> None:
    ## Attributes ----
    self.linkage = H.linkage
    self.BH = H.BH
    self.Z = H.Z
    self.H = H.H
    self.A = H.A
    self.nonzero = H.nonzero
    self.dA = H.dA
    self.nodes = H.nodes
    self.mode = H.mode
    self.leaves = H.leaves
    self.index = H.index
    self.R = H.R
    # Net ----
    self.path = H.plot_path
    self.areas = NET.struct_labels
    # Get regions and colors ----
    self.colregion = H.colregion
    self.colregion.get_regions()

  def aesthetic_ids(self, dA):
    ids = np.sort(np.unique(dA["id"].to_numpy()))
    if -1 in ids:
      ids = ids[1:]
      aids = np.arange(1, len(ids) + 1)
    else:
      aids = np.arange(1, len(ids) + 1)
    for i, id in enumerate(ids):
      dA.loc[dA["id"] == id, "id"] = aids[i].astype(str)
    dA["id"] = dA["id"].astype(int)

  def plot_measurements_mu(self, **kwargs):
    print("Plot Mu iterations")
    from pandas import DataFrame, concat
    # Create Data ----
    dF = DataFrame()
    # Concatenate over n and beta ----
    for i in np.arange(len(self.BH)):
      dF = concat(
        [
          dF,
          self.BH[i]
        ],
        ignore_index=True
      )
    dF.alpha = dF.alpha.to_numpy().astype(str)
    # Create figure ----
    g = sns.FacetGrid(
      dF, row="alpha",
      aspect=1, height=6
    )
    g.map_dataframe(
      sns.lineplot,
      x="K",
      y="mu",
      hue="beta"
    ).set(xscale="log")

  def plot_measurements_D(self, **kwargs):
    print("Plot D iterations")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.BH[0],
      x="K",
      y="D",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
    fig.tight_layout()

  def plot_measurements_X(self, **kwargs):
    print("Plot X iterations")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.BH[0],
      x="K",
      y="X",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
    fig.tight_layout()

  def plot_measurements_order_parameter(self, **kwargs):
    print("Plot order parameter iterations")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.BH[0],
      x="K",
      y="m",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
    fig.tight_layout()

  def plot_measurements_susceptibility(self, **kwargs):
    print("Plot susceptibility iterations")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.BH[0],
      x="K",
      y="xm",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
    fig.tight_layout()

  def plot_measurements_ntrees(self, **kwargs):
    print("Plot ntrees iterations")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.BH[0],
      x="K",
      y="ntrees",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
    fig.tight_layout()
    
  def skim_partition(self, partition):
    from collections import Counter
    fq = Counter(partition)
    for i in fq.keys():
      if fq[i] == 1: partition[partition == i] = -1
    new_partition = partition
    ndc = np.unique(partition[partition != -1])
    for i, c in enumerate(ndc):
      new_partition[partition == c] = i
    return new_partition
  
  def core_dendrogram(self, R, cmap_name="hls", remove_labels=False):
    print("Visualize node-community dendrogram!!!")
    from scipy.cluster import hierarchy
    import matplotlib.colors
    # Create figure ----
    for r in R:
      if r == 1: r += 1
      partition = hierarchy.cut_tree(self.Z, r).ravel()
      new_partition = self.skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      cm = sns.color_palette(cmap_name, len(unique_clusters_id))
      dlf_col = "#808080"
      ##
      D_leaf_colors = {}
      for i, _ in enumerate(self.colregion.labels[:self.nodes]):
        if new_partition[i] != -1:
          D_leaf_colors[i] = matplotlib.colors.to_hex(cm[new_partition[i]])
        else: D_leaf_colors[i] = dlf_col
      ##
      link_cols = {}
      for i, i12 in enumerate(self.Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(self.Z) else D_leaf_colors[x]
          for x in i12)
        link_cols[i+1+len(self.Z)] = c1 if c1 == c2 else dlf_col
      fig, _ = plt.subplots(1, 1)
      if ~remove_labels:
        hierarchy.dendrogram(
          self.Z,
          labels=self.colregion.labels[
            :self.nodes
          ],
          color_threshold=self.Z[self.nodes - r, 2],
          link_color_func = lambda k: link_cols[k]
        )
      else:
        hierarchy.dendrogram(
          self.Z,
          labels=False,
          color_threshold=self.Z[self.nodes - r, 2],
          link_color_func = lambda k: link_cols[k]
        )
      fig.set_figwidth(10)
      fig.set_figheight(7)

  def heatmap_pure(self, **kwargs):
    print("Visualize pure logFLN heatmap!!!")
    if "labels" in kwargs.keys():
      ids = kwargs["labels"]
      I, fq = sort_by_size(ids, self.nodes)
    else:
      I = np.arange(self.nodes, dtype=int)
      fq = {}
    # Transform FLNs ----
    W = self.R.copy()
    W[~self.nonzero] = np.nan
    W = W[I, :][:, I]
    # Configure labels ----
    labels = self.colregion.labels[I]
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
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(19)
    fig.set_figheight(15 * W.shape[0]/ self.nodes)
    sns.heatmap(
      W,
      cmap=sns.color_palette("viridis", as_cmap=True),
      xticklabels=labels[:self.nodes],
      yticklabels=labels,
      ax = ax
    )
    # Setting labels colors ----
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.xaxis.get_ticklabels()
      )
    ]
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.yaxis.get_ticklabels()
      )
    ]
    # Add black lines ----
    if "labels" in kwargs.keys():
      c = 0
      for key in fq:
        c += fq[key]
        if c < self.nodes:
          ax.vlines(
            c, ymin=0, ymax=self.nodes,
            linewidth=2,
            colors=["#C70039"]
          )
          ax.hlines(
            c, xmin=0, xmax=self.nodes,
            linewidth=2,
            colors=["#C70039"]
          )

  def heatmap_dendro(self):
    print("Visualize logFLN heatmap!!!")
    # Transform FLNs ----
    W = self.R.copy()
    # print(W)
    W[~self.nonzero] = np.nan
    # Get nodes ordering ----
    from scipy.cluster import hierarchy
    den_order = np.array(
      hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
    ).astype(int)
    W = W[den_order, :][:, den_order]
    # Configure labels ----
    labels = self.colregion.labels
    labels =  np.char.lower(labels[den_order].astype(str))
    rlabels = [
      str(re) for re in self.colregion.regions[
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
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(22)
    fig.set_figheight(15)
    sns.heatmap(
      W,
      xticklabels=labels,
      yticklabels=labels,
      ax = ax
    )
    # Setting labels colors ----
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.xaxis.get_ticklabels()
      )
    ]
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.yaxis.get_ticklabels()
      )
    ]

  def lcmap_pure(self, K, **kwargs):
    print("Visualize pure LC memberships!!!")
    if "labels" in kwargs.keys():
      ids = kwargs["labels"]
      I, fq = sort_by_size(ids, self.nodes)
      flag_fq = True
    else:
      I = np.arange(self.nodes, dtype=int)
      fq = {}
      flag_fq = False
    if "order" in kwargs.keys():
      I = kwargs["order"]
    for k in K:
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      dFLN["id"] =  cut_tree(
        self.H,
        n_clusters = k
      ).reshape(-1)
      minus_one_Dc(dFLN)
      self.aesthetic_ids(dFLN)
      keff = np.unique(
        dFLN["id"].to_numpy()
      ).shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      dFLN = dFLN[I, :][:, I]
      dFLN[dFLN == 0] = np.nan
      # dFLN = dFLN.T
      # Configure labels ----
      labels = self.colregion.labels[I]
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
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(19)
      fig.set_figheight(15 * dFLN.shape[0]/ self.nodes)
      sns.heatmap(
        dFLN,
        xticklabels=labels[:self.nodes],
        yticklabels=labels,
        cmap=sns.color_palette("hls", keff + 1),
        ax = ax
      )
      # Setting labels colors ----
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.xaxis.get_ticklabels()
        )
      ]
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.yaxis.get_ticklabels()
        )
      ]
      # Add black lines ----
      if flag_fq:
        c = 0
        for key in fq:
          c += fq[key]
          if c < self.nodes:
            ax.vlines(
              c, ymin=0, ymax=self.nodes,
              colors=["black"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              colors=["black"]
            )
    
  def lcmap_dendro(
    self, K, cmap_name="hls", remove_labels= False, **kwargs
  ):
    print("Visualize k LCs!!!")
    # K loop ----
    for k in K:
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      dFLN["id"] =  cut_tree(
        self.H,
        n_clusters = k
      ).reshape(-1)
      ##
      dFLN["source_label"] = labels[dFLN.source]
      dFLN["target_label"] = labels[dFLN.target]
      minus_one_Dc(dFLN)
      self.aesthetic_ids(dFLN)
      keff = np.unique(dFLN.id)
      keff = keff.shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(
        hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
      ).astype(int)
      dFLN = dFLN[den_order, :]
      dFLN = dFLN[:, den_order]
      dFLN[dFLN == 0] = np.nan
      dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
      # Configure labels ----
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = np.array([
        str(r).lower() for r in regions.AREA
      ])
      colors = regions.COLOR.loc[match(labels,rlabels)].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(18)
      fig.set_figheight(15)
      # Check colors with and without trees (-1) ---
      if -1 in dFLN:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        cmap_heatmap = sns.color_palette(cmap_name, keff)
      if ~remove_labels:
        plot = sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=labels,
          yticklabels=labels
        )
        if "font_size" in kwargs.keys():
          if kwargs["font_size"] > 0:
            plot.set_xticklabels(
              plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
            )
            plot.set_yticklabels(
              plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
            )
        # Setting labels colors ----
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.xaxis.get_ticklabels()
          )
        ]
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.yaxis.get_ticklabels()
          )
        ]
      else:
        plot = sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=False,
          yticklabels=False
        )
      
  def flatmap_dendro(self, NET, K, R, save=False, **kwargs):
    print("Plot single-linkage flatmap!!!")
    for i, kk in enumerate(K):
      # Get node ids ----
      from scipy.cluster.hierarchy import cut_tree
      ids = cut_tree(
        self.Z,
        n_clusters=R[i]
      ).reshape(-1)
      if "labels" in kwargs.keys(): ids = kwargs["labels"]
      # Start old-new mapping ---
      new_ids = {k : k for k in np.unique(ids)}
      # Look for isolated nodes ----
      from collections import Counter
      C = Counter(ids)
      C = [k for k in C if C[k] == 1]
      print("Number of isolated nodes:", len(C))
      # Create old-new transformation ----
      for k in C: new_ids[k] = -1
      ids = np.array([new_ids[k] for k in ids])
      ids = aesthetic_ids_vector(ids)
      F = FLATMAP(
        NET, self.colregion.regions.copy(), **kwargs
      )
      F.set_para(kk, R[i], ids)
      F.plot_flatmap(save=save)

  def plot_networx(self, r, rlabels, score="", cmap_name="", **kwargs):
    print("Draw networkx!!!")
    rlabels = self.skim_partition(rlabels)
    unique_labels = np.unique(rlabels)
    number_of_communities = unique_labels.shape[0]
    if -1 in unique_labels:
      save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
      color_map = [[]] * number_of_communities
      color_map[0] = [199/ 255.0, 0, 57/ 255.0]
      color_map[1:] = save_colors
    else:
      color_map = sns.color_palette(cmap_name, number_of_communities)
    color_dict = dict()
    for i, lab in enumerate(unique_labels):
      if lab != -1: color_dict[lab] = color_map[i]
      else: color_dict[-1] = "#808080"
    node_colors = [
      color_dict[lab] for lab in rlabels
    ]
    G = nx.from_numpy_array(
      self.A, create_using=nx.DiGraph
    )
    Ainv = self.A.copy()
    Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
    Ginv = nx.from_numpy_array(
      Ainv, create_using=nx.DiGraph
    )
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    pos = nx.kamada_kawai_layout(Ginv)
    pos = nx.spring_layout(
      G, pos=pos, iterations=5, seed=212
    )
    nx.draw_networkx(
      G,
      pos=pos,
      node_color=node_colors,
      connectionstyle="arc3,rad=0.1",
      ax=ax, **kwargs
    )
    fig.tight_layout()
  
  def plot_networx_link_communities(self, K, score="", cmap_name="hls", **kwargs):
    print("Draw networkx link communities!!!")
    dA = self.dA.copy()
    from scipy.cluster.hierarchy import cut_tree
    for k in K:
      labels = cut_tree(self.H, k).ravel()
      dA["id"] = labels
      minus_one_Dc(dA)
      self.aesthetic_ids(dA)
      labels = dA.id.to_numpy()
      labels[labels > 0] = labels[labels > 0] - 1
      unique_labels = np.unique(labels)
      number_of_communities = unique_labels.shape[0]
      if -1 in unique_labels:
        save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
        color_map = [[]] * number_of_communities
        color_map[0] = [199/ 255.0, 0, 57/ 255.0]
        color_map[1:] = save_colors
      else:
        color_map = sns.color_palette(cmap_name, number_of_communities)
      color_dict = dict()
      for i, lab in enumerate(unique_labels):
        if lab != -1: color_dict[lab] = color_map[i]
        else: color_dict[-1] = "#808080"
      edge_colors = [
        color_dict[lab] for lab in labels
      ]
      G = nx.from_numpy_array(
        self.A, create_using=nx.DiGraph
      )
      Ainv = self.A.copy()
      Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
      Ginv = nx.from_numpy_array(
        Ainv, create_using=nx.DiGraph
      )
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      pos = nx.kamada_kawai_layout(Ginv)
      pos = nx.spring_layout(
        G, pos=pos, iterations=5, seed=212
      )
      nx.draw_networkx(
        G, pos=pos,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.1",
        ax=ax, **kwargs
      )
      fig.tight_layout()