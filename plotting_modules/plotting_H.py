# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
# Personal libs ----
from various.network_tools import *
from modules.flatmap import FLATMAP

class Plot_H:
  def __init__(self, NET, H, sln=False) -> None:
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
    self.entropy = H.entropy
    self.R = H.R
    ##
    if sln:
      self.sln = NET.sln
      self.supra = NET.supra
      self.infra = NET.infra
    # Net ----
    self.path = H.plot_path
    self.areas = NET.struct_labels
    # Get regions and colors ----
    self.colregion = H.colregion
    self.colregion.get_regions()

  def Mu_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize MU with plotly!!!")
      from pandas import DataFrame, concat
      # Create Data ----
      dF = DataFrame()
      # Concatenate over n and beta ----
      for i in np.arange(len(self.BH)):
        dF = concat([dF, self.BH[i]])
      fig = px.line(
        dF,
        x = "K",
        y = "mu",
        color = "beta",
        facet_col = "alpha",
        facet_col_wrap = 2,
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path,  "Features")
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Mu_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Mu with plotly")
    
  def plot_newick_R(self, tree_newick, weighted=False, on=True):
    if on:
      print("Plot tree in Newick format from R!!!")
      import subprocess
      # Arrange path ----
      plot_path = join(self.path, "NEWICK")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      if not weighted:
        subprocess.run(["Rscript", "R/plot_newick_tree.R", tree_newick, join(plot_path, "tree_newick.png")])
      else:
        subprocess.run(["Rscript", "R/plot_newick_tree.R", tree_newick, join(plot_path, "tree_newick_H.png")])
    else:
      print("No tree in Newick format")

  def plot_measurements_mu(self, on=False, **kwargs):
    if on:
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
          ], ignore_index=True
        )
      dF.alpha = dF.alpha.to_numpy().astype(str)
      dF.beta = dF.beta.to_numpy().astype(str)
      # Create figure ----
      g = sns.FacetGrid(
        dF, col="alpha",
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="K",
        y="mu",
        hue="beta"
      ).set(xscale="log")
      g.add_legend()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "Mu_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No Mu iterations")

  def Entropy_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize Entropy with plotly!!!")
      # Create data ----
      dim = self.link_entropy.shape[1]
      k = np.arange(dim, 0, -1)
      data = pd.DataFrame(
        {
        "K" : np.hstack([k, k]),
        "S" : self.link_entropy.ravel(),
        "c" : ["Sh"] * dim + ["Sv"] * dim
        }
      )
      # Create figure ----
      fig = px.line(
        data,
        x = "K",
        y = "S",
        color = "c",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Entropy_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Entropy with plotly")

  def Entropy_H_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize Entropy with plotly!!!")
      # Create data ----
      dim = self.link_entropy_H.shape[1]
      k = np.arange(dim, 0, -1)
      data = pd.DataFrame(
        {
        "K" : np.hstack([k, k]),
        "S" : self.link_entropy_H.ravel(),
        "c" : ["Sh"] * dim + ["Sv"] * dim
        }
      )
      # Create figure ----
      fig = px.line(
        data,
        x = "K",
        y = "S",
        color = "c",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Entropy_H_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Entropy with plotly")

  def plot_measurements_Entropy(self, on=True):
    if on:
      print("Visualize Entropy iterations!!!")
      # Create data ----
      dim = self.entropy[0].shape[1]
      print(f"Levels node hierarchy: {dim}")
      data = pd.DataFrame(
        {
          "S" : np.hstack([self.entropy[0].ravel(), self.entropy[1].ravel()]),
          "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
          "c" : ["node_hierarchy"] * 2 * dim + ["node_hierarchy_H"] * 2 * dim,
          "level" : list(np.arange(dim, 0, -1)) * 4
        }
      )
      dim = self.entropy[2].shape[1]
      print(f"Levels link hierarchy: {dim}")
      data = pd.concat(
        [
          data,
          pd.DataFrame(
            {
              "S" : np.hstack([self.entropy[2].ravel(), self.entropy[3].ravel()]),
              "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
              "c" : ["link_hierarchy"] * 2 * dim + ["link_hierarchy_H"] * 2 * dim,
              "level" : list(np.arange(dim, 0, -1)) * 4
            }
          )
        ], ignore_index=True
      )
      mx = data.iloc[
        data.groupby(["c", "dir"])["S"].transform("idxmax").drop_duplicates(keep="first").to_numpy()
      ].sort_values("c", ascending=False)
      print(mx)
      # Create figure ----
      g = sns.FacetGrid(
        data=data,
        col = "c",
        hue = "dir",
        col_wrap=2,
        sharex=False,
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="level",
        y="S"
      )#.set(xscale="log")
      g.add_legend()
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        join(
          plot_path, "Entropy_levels.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No Entropy iterations")

  def D_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize D with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "D",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "D_{}.html".format(self.linkage)
        )
      )
    else:
      print("No D with plotly")

  def plot_measurements_D(self, on=False, **kwargs):
    if on:
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
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "D_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No D iterations")

  def plot_measurements_S(self, on=False, **kwargs):
    if on:
      print("Plot S iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="S",
        ax=ax
      )
      ax.set_ylabel(r"$H_{L}$")
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "S_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No S iterations")

  def plot_measurements_SD(self, on=False, **kwargs):
    if on:
      print("Plot SD iterations")
      # Create figure ----
      if "SD" not in self.BH[0].columns:
        self.BH[0]["SD"] = (self.BH[0].D / np.nansum(self.BH[0].D)) * (self.BH[0].S / np.nansum(self.BH[0].S))
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="SD",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "SD_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No SD iterations")

  def X_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize X with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "X",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "X_{}.html".format(self.linkage)
        )
      )
    else:
      print("No X with plotly")

  def plot_measurements_X(self, on=False, **kwargs):
    if on:
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
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "X_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No X iterations")
  
  def order_parameter_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize the order parameter with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "m",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "mt_{}.html".format(self.linkage)
        )
      )
    else:
      print("No the order parameter with plotly")

  def plot_measurements_order_parameter(self, on=False, **kwargs):
    if on:
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
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "order_parameter_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No order parameter iterations")
  
  def susceptibility_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize xm with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "xm",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "xmt_{}.html".format(self.linkage)
        )
      )
    else:
      print("No xm(t) with plotly")

  def plot_measurements_susceptibility(self, on=False, **kwargs):
    if on:
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
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "xm_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No susceptibility iterations")

  def ntrees_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize ntree with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "ntrees",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "ntrees_{}.html".format(self.linkage)
        )
      )
    else:
      print("No ntrees with plotly")

  def plot_measurements_ntrees(self, on=False, **kwargs):
    if on:
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
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "ntrees_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No ntrees iterations")
  
  def core_dendrogram(self, R : list, score="", cmap_name="hls", remove_labels=False, on=False, **kwargs):
    if on:
      print("Visualize node-community dendrogram!!!")
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      from scipy.cluster import hierarchy
      import matplotlib.colors
      # Create figure ----
      for r in R:
        if r == 1:
          r += 1
          sname = "fake"
        else: sname = ""
        #
        partition = hierarchy.cut_tree(self.Z, r).ravel()
        new_partition = skim_partition(partition)
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
        if not remove_labels:
          hierarchy.dendrogram(
            self.Z,
            labels=self.colregion.labels[:self.nodes],
            color_threshold=self.Z[self.nodes - r, 2],
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, **kwargs
          )
        else:
          hierarchy.dendrogram(
            self.Z,
            no_labels=True,
            color_threshold=self.Z[self.nodes - r, 2],
            link_color_func = lambda k: link_cols[k]
          )
        fig.set_figwidth(10)
        fig.set_figheight(7)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "core_dendrogram_{}_{}{}{}.png".format(self.linkage, r, score, sname)
          ),
          dpi=500
        )
        plt.close()
    else:
      print("No node-community dendrogram")

  def heatmap_pure(self, r, R, score="", linewidth=1.5, on=True, **kwargs):
    if on:
      print("Visualize pure logFLN heatmap!!!")
      if "labels" in kwargs.keys():
        ids = kwargs["labels"]
        I, fq = sort_by_size(ids, self.nodes)
      else:
        I = np.arange(self.nodes, dtype=int)
        fq = {}
      # Transform FLNs ----
      W = R.copy()
      nonzero = (W != 0)
      W[~nonzero] = np.nan
      W = W[I, :][:, I]
      # Configure labels ----
      labels = self.colregion.labels[I]
      labels = [str(r) for r in labels]
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
              linewidth=linewidth,
              colors=["#C70039"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              linewidth=linewidth,
              colors=["#C70039"]
            )
     # Arrange path ----
      plot_path = os.path.join(self.path, "Heatmap_pure")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "heatmap{}_{}.png".format(score, r)
        ),
        dpi = 300
      )
      plt.close()
    else:
      print("No pure logFLN heat map")
  
  # def heatmap_pure_sln(self, on=True, **kwargs):
  #   if on:
  #     print("Visualize pure sln heatmap!!!")
  #     if "labels" in kwargs.keys():
  #       ids = kwargs["labels"]
  #       I, fq = sort_by_size(ids, self.nodes)
  #     else:
  #       I = np.arange(self.nodes, dtype=int)
  #       fq = {}
  #     if "name" in kwargs.keys():
  #       name = kwargs["name"]
  #     else: name = ""
  #     if "order" in kwargs.keys():
  #       I = kwargs["order"]
  #     # Transform FLNs ----
  #     w = self.A.copy()
  #     W = self.sln.copy()
  #     W[w == 0] = np.nan
  #     W = W[I, :][:, I]
  #     # Configure labels ----
  #     labels = self.colregion.labels[I]
  #     rlabels = [
  #       str(r) for r in self.colregion.regions[
  #         "AREA"
  #       ]
  #     ]
  #     colors = self.colregion.regions.loc[
  #       match(
  #         labels,
  #         rlabels
  #       ),
  #       "COLOR"
  #     ].to_numpy()
  #     # Create figure ----
  #     fig, ax = plt.subplots(1, 1)
  #     fig.set_figwidth(19)
  #     fig.set_figheight(15 * W.shape[0]/ self.nodes)
  #     cmp = sns.diverging_palette(30, 250,  l=65, center="dark", as_cmap=True)
  #     sns.heatmap(
  #       W,
  #       xticklabels=labels[:self.nodes],
  #       yticklabels=labels,
  #       center=0.5,
  #       cmap=cmp,
  #       ax = ax
  #     )
  #     # Setting labels colors ----
  #     [t.set_color(i) for i,t in
  #       zip(
  #         colors,
  #         ax.xaxis.get_ticklabels()
  #       )
  #     ]
  #     [t.set_color(i) for i,t in
  #       zip(
  #         colors,
  #         ax.yaxis.get_ticklabels()
  #       )
  #     ]
  #     # Add black lines ----
  #     if "labels" in kwargs.keys():
  #       c = 0
  #       for key in fq:
  #         c += fq[key]
  #         if c < self.nodes:
  #           ax.vlines(
  #             c, ymin=0, ymax=self.nodes,
  #             linewidth=2,
  #             colors=["gold"]
  #           )
  #           ax.hlines(
  #             c, xmin=0, xmax=self.nodes,
  #             linewidth=2, 
  #             colors=["gold"]
  #           )
  #     # Arrange path ----
  #     plot_path = os.path.join(self.path, "Heatmap_pure_sln")
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save plot ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, "{}.png".format(name)
  #       ),
  #       dpi = 300
  #     )
  #   else:
  #     print("No pure sln heatmap")

  def heatmap_dendro(self, r, R, feature="", score="", linewidth=1.5, on=True, **kwargs):
    if on:
      print("Visualize logFLN heatmap!!!")
      # Transform FLNs ----
      W = R.copy()
      W[W == 0] = np.nan
      W[W == -np.Inf] = np.nan
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(
        hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
      ).astype(int)
      memberships = hierarchy.cut_tree(self.Z, r).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
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
      fig.set_figwidth(18)
      fig.set_figheight(15)
      sns.heatmap(
        W,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
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
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Heatmap_single"
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
      print("No logFLN heat map")

  # def heatmap_dendro_sln(self, r, on=True):
  #   if on:
  #     print("Visualize SLN heatmap!!!")
  #     # Transform FLNs ----
  #     w = self.A.copy()
  #     W = self.sln.copy()
  #     W[w == 0] = np.nan
  #     # Get nodes ordering ----
  #     from scipy.cluster import hierarchy
  #     den_order = np.array(
  #       hierarchy.dendrogram(self.Z)["ivl"]
  #     ).astype(int)
  #     W = W[den_order, :]
  #     W = W[:, den_order]
  #     # W = W.T
  #     # Configure labels ----
  #     labels = self.colregion.labels
  #     labels =  np.char.lower(labels[den_order].astype(str))
  #     rlabels = [
  #       str(r) for r in self.colregion.regions[
  #         "AREA"
  #       ]
  #     ]
  #     colors = self.colregion.regions.loc[
  #       match(
  #         labels,
  #         rlabels
  #       ),
  #       "COLOR"
  #     ].to_numpy()
  #     # Create figure ----
  #     fig, ax = plt.subplots(1, 1)
  #     fig.set_figwidth(22)
  #     fig.set_figheight(15)
  #     cmp = sns.diverging_palette(30, 250,  l=65, center="dark", as_cmap=True)
  #     sns.heatmap(
  #       W,
  #       center=0.5,
  #       xticklabels=labels,
  #       yticklabels=labels,
  #       cmap=cmp,
  #       ax = ax
  #     )
  #     # Setting labels colors ----
  #     [t.set_color(i) for i,t in
  #       zip(
  #         colors,
  #         ax.xaxis.get_ticklabels()
  #       )
  #     ]
  #     [t.set_color(i) for i,t in
  #       zip(
  #         colors,
  #         ax.yaxis.get_ticklabels()
  #       )
  #     ]
  #     # Arrange path ----
  #     plot_path = os.path.join(self.path, "Heatmap_sln")
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save plot ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, f"{r}.png"
  #       ),
  #       dpi = 300
  #     )
  #   else:
  #     print("No heatmap sln")

  def lcmap_pure(self, K, score="", cmap_name="husl", linewidth=1.5, undirected=False, on=True, **kwargs):
    if on:
      print("Visualize pure LC memberships!!!")
      # Get elemets from colregion ----
      labels = self.colregion.labels
      regions = self.colregion.regions
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
        if not undirected:
          dFLN["id"] =  cut_tree(
            self.H,
            n_clusters = k
          ).ravel()
        else:
          dFLN["id"] = np.tile(cut_tree(
            self.H,
            n_clusters = k
          ).ravel(), 2)
        minus_one_Dc(dFLN, undirected)
        aesthetic_ids(dFLN)
        keff = np.unique(
          dFLN["id"].to_numpy()
        ).shape[0]
        # Transform dFLN to Adj ----
        dFLN = df2adj(dFLN, var="id")
        dFLN = dFLN[I, :][:, I]
        dFLN[dFLN == 0] = np.nan
        dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
        # dFLN = dFLN.T
        # Configure labels ----
        labels = self.colregion.labels[I]
        labels = [str(r) for r in labels]
        rlabels = [str(r) for r in regions.AREA]
        colors = regions.COLOR.loc[match(labels, rlabels)].to_numpy()
        # Create figure ----
        fig, ax = plt.subplots(1, 1)
        # Check colors with and without trees (-1) ---
        if -1 in dFLN:
          save_colors = sns.color_palette(cmap_name, keff - 1)
          cmap_heatmap = [[]] * keff
          cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
          cmap_heatmap[1:] = save_colors
        else:
          cmap_heatmap = sns.color_palette(cmap_name, keff)
        fig.set_figwidth(19)
        fig.set_figheight(15 * dFLN.shape[0]/ self.nodes)
        sns.heatmap(
          dFLN,
          xticklabels=labels[:self.nodes],
          yticklabels=labels,
          cmap=cmap_heatmap,
          annot_kws = {"size" : 12},
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
                colors=["black"], linewidth = linewidth
              )
              ax.hlines(
                c, xmin=0, xmax=self.nodes,
                colors=["black"], linewidth = linewidth
              )
        # Arrange path ----
        plot_path = os.path.join(self.path, "Matrix_pure")
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
        plt.close()
    else:
      print("No pure logFLN heat map")
    
  def plot_histoscatter_linkcomm(self, k, ncol=1, on=False, **kwargs):
    if on:
      print("Plot histoscatter linkcomm for A!!!")
      dA = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      dA["id"] =  cut_tree(
        self.H,
        n_clusters = k
      ).reshape(-1)
      minus_one_Dc(dA)
      aesthetic_ids(dA)
      dA.weight = np.log(dA.weight)
      dA.id = dA.id.astype(str)
      dA = dA.sort_values(by="id", ignore_index=True)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        hue="id",
        element="poly",
        stat="density",
        alpha=0.5,
        fill=False,
        common_norm=True,
        ax=ax
      )
      ax.set_yscale("log")
      sns.move_legend(ax, "upper left", bbox_to_anchor=(1.03, 1), ncol=ncol)
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "histoscatter_fln_link.png"
        ),
        dpi=300
      )
    else: print("No histoscatter linkcomm")

  def lcmap_dendro(
    self, K, R, score="", cmap_name="hls",
    link_com_list=False, remove_labels=False, linewidth=1.5, undirected=False, on=False, **kwargs
  ):
    if on:
      print("Visualize k LCs!!!")
      # K loop ----
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Matrix_single"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      if not undirected:
        dFLN["id"] =  cut_tree(
          self.H,
          n_clusters = K
        ).ravel()
      else:
        dFLN["id"] =  np.tile(cut_tree(
          self.H,
          n_clusters = K
        ).ravel(), 2)
      ##
      dFLN["source_label"] = labels[dFLN.source]
      dFLN["target_label"] = labels[dFLN.target]
      # Print link community list ----
      if link_com_list:
        dFLN.sort_values(by="id").to_csv(
          f"{plot_path}/link_community_list_{K}.csv"
        )
      ##
      minus_one_Dc(dFLN, undirected)
      aesthetic_ids(dFLN)
      keff = np.unique(dFLN.id)
      keff = keff.shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      memberships = hierarchy.cut_tree(self.Z, R).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      #
      dFLN = dFLN[den_order, :]
      dFLN = dFLN[:, den_order]
      dFLN[dFLN == 0] = np.nan
      dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
      # Configure labels ----
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = np.array([str(r).lower() for r in regions.AREA])
      colors = regions.COLOR.loc[match( labels, rlabels)].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      # Check colors with and without trees (-1) ---
      if -1 in dFLN:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        cmap_heatmap = sns.color_palette(cmap_name, keff)
      if not remove_labels:
        plot = sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=labels,
          yticklabels=labels,
          ax=ax
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
        [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
        [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]
      else:
        sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=False,
          yticklabels=False,
          ax=ax
        )
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
      fig.set_figwidth(18)
      fig.set_figheight(15)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "{}{}.png".format(K, score)
        )
      )
      plt.close()
    else:
      print("No k LCs")

  # def link_community_sln(self, K, on=False):
  #   if on:
  #     print("SLN link communities!!!")
  #     for k in K:
  #       # Get labels ----
  #       labels = self.colregion.labels
  #       regions = self.colregion.regions.copy()
  #       regions = regions.loc[np.isin(regions.AREA, labels)]
  #       regions.AREA = regions.AREA.to_numpy().astype("<U7")
  #       regions = regions.set_index(regions.AREA)
  #       regions = regions.reindex(labels).set_index(np.arange(regions.shape[0]))
  #       # Arrange path ----
  #       plot_path = os.path.join(self.path, "linkc_sln")
  #       # Crate path ----
  #       Path(
  #         plot_path
  #       ).mkdir(exist_ok=True, parents=True)
  #       # FLN to dataframe and filter FLN = 0 ----
  #       dFLN = self.dA.copy()
  #       dFLN["W1"] = -np.log(dFLN.weight)
  #       dFLN["W2"] = np.log(dFLN.weight) + 7
  #       # Add id with aesthethis ----
  #       from scipy.cluster.hierarchy import cut_tree
  #       dFLN["id"] =  cut_tree(
  #         self.H,
  #         n_clusters = k
  #       ).reshape(-1)
  #       ##
  #       dFLN["source_label"] = labels[dFLN.source]
  #       dFLN["target_label"] = labels[dFLN.target]
  #       ##
  #       self.minus_one_Dc(dFLN)
  #       self.aesthetic_ids(dFLN)
  #       dFLN = dFLN.loc[dFLN.id != -1]
  #       IDS = dFLN.id.to_numpy()
  #       plt.close()
  #       for id in np.unique(IDS):
  #         G1 = nx.DiGraph()
  #         subnet = dFLN.loc[dFLN.id == id]
  #         edgelist = zip(
  #           subnet.source.to_numpy(),
  #           subnet.target.to_numpy(),
  #           subnet.W1.to_numpy(),
  #           subnet.sln.to_numpy()
  #         )
  #         for e in edgelist:
  #           src = e[0]
  #           dst = e[1]
  #           w = e[2]
  #           sln = e[3]
  #           G1.add_node(labels[src], color=regions.COLOR.iloc[src])
  #           G1.add_node(labels[dst], color=regions.COLOR.iloc[dst])
  #           if sln >= 0.5:
  #             G1.add_edge(labels[src], labels[dst], value=w, color="blue")
  #           else:
  #             G1.add_edge(labels[src], labels[dst], value=w, color="red")
  #         G2 = nx.DiGraph()
  #         edgelist = zip(
  #           subnet.source.to_numpy(),
  #           subnet.target.to_numpy(),
  #           subnet.W2.to_numpy(),
  #           subnet.sln.to_numpy()
  #         )
  #         for e in edgelist:
  #           src = e[0]
  #           dst = e[1]
  #           w = e[2]
  #           sln = e[3]
  #           G2.add_node(labels[src], color=regions.COLOR.iloc[src])
  #           G2.add_node(labels[dst], color=regions.COLOR.iloc[dst])
  #           if sln >= 0.5:
  #             G2.add_edge(labels[src], labels[dst], value=w, color="blue")
  #           else:
  #             G2.add_edge(labels[src], labels[dst], value=w, color="red")
  #         pos = nx.kamada_kawai_layout(G1)
  #         pos = nx.spring_layout(G2, pos=pos, seed=8998, iterations=6, k=2)
  #         nodes = G1.nodes()
  #         edges = G1.edges()
  #         edge_color = [G1[u][v]['color'] for u, v in edges]
  #         nx.draw_networkx_edges(
  #           G1, pos=pos, edge_color=edge_color, alpha=0.4, connectionstyle="arc3,rad=0.1"
  #         )
  #         rlabels = {}
  #         for c in nodes:
  #           rlabels[c] = c
  #         nx.draw_networkx_labels(G1, pos, labels=rlabels, font_size=10)
  #         # Save plot ----
  #         plt.savefig(
  #           os.path.join(
  #             plot_path, "{}_{}.png".format(k, id)
  #           ),
  #           dpi=200
  #         )
  #         plt.close()
  #     else: print("No sln link communities")
      
  def flatmap_dendro(self, NET, K, R, on=True, **kwargs):
    if on:
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
        F.plot_flatmap()
    else:
      print("No single-linkage flatmap")

  #** Average linkage ----
  def lcmap_average(self, K, R, WSBM, on=True):
    if on:
      print("Visualize k LCs average!!!")
      from scipy.cluster.hierarchy import cut_tree
      # Add id with aesthethis ----
      for k in K:
        for r in R:
          dFLN = self.dA.copy()
          dFLN["id"] =  cut_tree(
            self.H,
            n_clusters = k
          ).reshape(-1)
          minus_one_Dc(dFLN)
          aesthetic_ids(dFLN)
          # Transform dFLN to Adj ----
          dFLN = df2adj(dFLN, var="id")
          # Get memberships ----
          ids = WSBM.labels.loc[
            (WSBM.labels["K"] == k) &
            (WSBM.labels["R"] == r),
            "labels"
          ].to_numpy()
          ## Sorted memberships ----
          I, fq = sort_by_size(ids, self.nodes)
          dFLN = dFLN[I, :][:, I]
          dFLN[dFLN == 0] = np.nan
          # dFLN = dFLN.T
          # Configure labels ----
          labels = self.colregion.labels
          labels =  labels[I]
          rlabels = [
            str(ri) for ri in self.colregion.regions[
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
          ## Total number of LCs ----
          keff = np.unique(dFLN, equal_nan=True)
          keff = len(keff)
          # Create figure ----
          fig, ax = plt.subplots(1, 1)
          fig.set_figwidth(18)
          fig.set_figheight(15)
          sns.heatmap(
            dFLN,
            cmap=sns.color_palette("hls", keff),
            cbar=False,
            xticklabels=labels,
            yticklabels=labels
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
          # Arrange path ----
          plot_path = os.path.join(
            self.path, "Matrix_average", "K_{}".format(k)
          )
          # Crate path ----
          Path(
            plot_path
          ).mkdir(exist_ok=True, parents=True)
          # Save plot ----
          plt.savefig(
            os.path.join(
              plot_path, "{}.png".format(r)
            )
          )
    else:
      print("No k LCs")

  def flatmap_average(self, NET, K, R, WSBM, on=True):
    if on:
      print("Plotting average-linkage flatmap!!!")
      fmap = FLATMAP(NET, self.colregion.regions.copy())
      for k in K:
        for r in R:
          print("Flatmap K: {}, R: {}".format(k, r))
          # Get memberships ----
          ids = WSBM.labels.loc[
            (WSBM.labels["K"] == k) &
            (WSBM.labels["R"] == r),
            "labels"
          ].to_numpy()
          fmap.set_para(k, r, ids)
          fmap.plot_flatmap()
    else:
      print("No average-linkage flatmap")
  
  def flatmap_labels(self, r, labels, on=True, **kwargs):
    if on:
      print("Plotting average-linkage flatmap!!!")
      labels = aesthetic_ids_vector(labels)
      fmap = FLATMAP(
        self.NET, self.colregion.regions.copy(), **kwargs
      )
      print("Flatmap with labels")
      fmap.set_para(r, r, labels)
      fmap.plot_flatmap()
    else:
      print("No average-linkage flatmap")

  def plot_networx(
    self, r, rlabels, score="", cmap_name="", on=False,
    remove_labels = False, **kwargs
  ):
    if on:
      print("Draw networkx!!!")
      rlabels = skim_partition(rlabels)
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
      if not remove_labels:
        nx.draw_networkx(
          G,
          pos=pos,
          node_color=node_colors,
          connectionstyle="arc3,rad=0.1",
          ax=ax, **kwargs
        )
      else:
        nx.draw_networkx(
          G,
          pos=pos,
          node_color=node_colors,
          connectionstyle="arc3,rad=0.1",
          ax=ax, with_labels=False
        )
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "netwokx"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "{}{}.png".format(r, score)
        )
      )
      plt.close()
    else: print("No networkx")
  
  def plot_networx_link_communities(self, K, score="", cmap_name="hls", on=False, **kwargs):
    if on:
      print("Draw networkx link communities!!!")
      dA = self.dA.copy()
      from scipy.cluster.hierarchy import cut_tree
      for k in K:
        labels = cut_tree(self.H, k).ravel()
        dA["id"] = labels
        minus_one_Dc(dA)
        aesthetic_ids(dA)
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
        # Arrange path ----
        plot_path = os.path.join(
          self.path, "netwokx_link_comm"
        )
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "{}{}.png".format(k, score)
          )
        )
        plt.close()
    else: print("No networkx link communities")