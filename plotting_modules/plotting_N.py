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
from modules.hierarmerge import Hierarchy

class Plot_N:
  def __init__(self, NET, H : Hierarchy) -> None:
    # From net ----
    self.path = NET.plot_path
    # from Hierarchy ----
    self.nodes = H.nodes
    self.linkage = H.linkage
    self.A = H.A
    self.dA = H.dA
    self.D = H.D
    self.BH = H.BH
    self.R = H.R
    self.index = H.index
    # Data transformed ----
    self.aik = H.source_sim_matrix
    self.aki = H.target_sim_matrix

  def normal(self, mean, std, color="black"):
    from scipy import stats
    x = np.linspace(mean - (4 * std), mean + (4 * std), 200)
    p = stats.norm.pdf(x, mean, std)
    plt.plot(x, p, color, linewidth=2)


  def A_vs_dis(self, A, s=1, name="weight", on=True, **kwargs):
    if on:
      print("Plot {} vs dist!!!".format(name))
      # Get data ----
      dA = adj2df(A.copy())
      dD = adj2df(
        self.D.copy()[:, :self.nodes]
      )
      # Get No connections ---
      zeros = dA["weight"] == 0
      # Elllminate zeros ---
      dA = dA.loc[~zeros]
      dD = dD.loc[~zeros]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "weight" : np.log(dA["weight"]),
          "dist" : dD["weight"]
        }
      )
      # Get slop and p-value ----
      model = linear_fit(
        data["dist"].to_numpy(),
        data["weight"].to_numpy()
      )
      x = np.linspace(
        np.min(self.D[self.D > 0]),
        np.max(self.D),
        100
      )
      # Create figure ----
      _ , ax = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="dist",
        y="weight",
        s=s
      )
      if "reg" in kwargs.keys():
        # x_st = (x - np.mean(x)) / np.std(x)
        if kwargs["reg"]:
          sns.lineplot(
            x=x,
            y=model.predict(x.reshape(-1, 1)),
            color="r"
          )
      ax.text(
        0.5, 1.05,
        s = "m={:.4f}".format(
          model.coef_[0]
        ),
        ha='center', va='center',
        transform=ax.transAxes
      )
      plt.ylabel("log(FLN)")
      plt.xlabel("dist [mm]")
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
          plot_path, "{}_vs_dist.png".format(name)
        ),
        dpi=300
      )
    else:
      print("No {} vs dist".format(name))

  def histogram_dist(self, on=True):
    if on:
      print("Plot distance histogram!!!")
      # Transform FLN to DataFrame ----
      dD = adj2df(
        self.D[:, :self.nodes].copy()
      )
      dD = dD.loc[dD.source > dD.target]
      # Create figure
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dD,
        x="weight",
        stat="density"
      )
      plt.xlabel("distance [mm]")
      plt.ylabel(r"$q(d)\,(mm^{-1})$")
      fig.tight_layout()
      # self.normal(
      #   dD["weight"].to_numpy().mean(),
      #   dD["weight"].to_numpy().std()
      # )
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
          plot_path, "dist_hist.png"
        ),
        dpi=300
      )
    else:
      print("No histogram")

  def projection_probability(
    self, C, bins=20, on=True, **kwargs
  ):
    if on:
      print("Plot neuron-distance freq!!!")
      # Distance range ----
      _, x, y = range_and_probs_from_DC(
        self.D, C, self.nodes, bins
      )
      from pandas import DataFrame
      y = np.exp(y)
      _, x_range_model, _, _, model = predicted_D_frequency(
        self.D, C, self.nodes, bins, **kwargs
      )
      y_pred = 0
      y_pred = model.predict(
        x.reshape(-1, 1)
      )
      y_pred = np.exp(y_pred)
      data = DataFrame(
        {
          "dist" : np.round(x, 1).astype(str),
          "prob" : y,
          "pred" : y_pred
        }
      )
      data = data.loc[
        data["prob"] > 0
      ]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      #  Plot data ----
      cmp = sns.color_palette("deep")
      sns.barplot(
        data=data,
        x="dist",
        y="prob",
        color=cmp[0],
        alpha=0.8,
        ax=ax
      )
      sns.scatterplot(
        data=data,
        x="dist",
        y="prob",
        color="black",
        ax=ax
      )
      # Plot prediction ----
      ## Plot pred ----
      label="linear model"
      sns.lineplot(
        data=data,
        linewidth=2,
        x="dist",
        y='pred',
        color="r",
        label=label,
        ax=ax
      )
      ax.text(
          0.5, 1.05,
          s="linear coeff: {:.5f}".format(
            model.coef_[0]
          ),
          ha='center', va='center',
          transform=ax.transAxes
      )
      ax.set_yscale('log')
      ax.legend()
      # Aesthetics ----
      ax.set_xlabel("dist [mm]")
      ax.set_ylabel("p(d) [1/mm]")
      ## Fig size ----
      fig.set_figheight(6)
      fig.set_figwidth(8)
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features", "projection_p"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      if "reg" in kwargs.keys():
        if kwargs["reg"] == "poly":
          plt.savefig(
            os.path.join(
              plot_path, "poly_bin_{}_deg_{}.png".format(
                bins, kwargs["deg"]
              )
            ),
            dpi=200
          )
        elif kwargs["reg"] == "piece_poly":
          plt.savefig(
            os.path.join(
              plot_path, "piece_poly_bin_{}.png".format(bins)
            ),
            dpi=200
          )
        elif kwargs["reg"] == "gp":
          plt.savefig(
            os.path.join(
              plot_path, "gp_bin_{}.png".format(bins)
            ),
            dpi=200
          )
      else:
        plt.savefig(
          os.path.join(
            plot_path, "bin_{}.png".format(bins)
          ),
          dpi=200
        )
    else:
      print("No histogram")

  def histogram_weight(self, feature="", on=True):
    if on:
      print("Plot weight histogram!!!")
      # Transform FLN to DataFrame ----
      dA = self.dA.copy()
      # Transform FLN to weights ----
      dA["weight"] = np.log(dA["weight"])
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        stat="density"
      )
      plt.xlabel(r"$\log(FLN)$")
      plt.ylabel(r"density $([\log(FLN)]^{-1})$")
      fig.tight_layout()
      # self.normal(
      #   dA["weight"].to_numpy().mean(),
      #   dA["weight"].to_numpy().std()
      # )
      # Arrange path ----
      plot_path = os.path.join(
        self.path,"Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "weight_histo.png"
        ),
        dpi=300
      )
    else:
      print("No histogram")

  def get_color_ids(self, dA, palette="husl"):
    ids = np.sort(
      np.unique(dA["id"].to_numpy())
    )
    n_ids = len(ids)
    cmap = sns.color_palette(palette, n_ids)
    colors = {}
    for i in np.arange(n_ids):
      colors[ids[i]] = cmap[i]
    return colors
  
  def plot_akis(self, D, s=1, on=True):
    if on:
      from scipy.stats import pearsonr, linregress
      print("Plot similarity plots and distance!!!")
      # Transform data to dataframe ----
      dD = adj2df(
        D.copy()[:self.nodes, :self.nodes]
      )
      daki = self.aki.copy()
      daki[np.isnan(daki)] = np.nanmin(daki) - 1
      daki = adj2df(daki)
      daki = daki.loc[
        daki["source"] < daki["target"], "weight"
      ].to_numpy().ravel()
      daik = self.aik.copy()
      daik[np.isnan(daik)] = np.nanmin(daik) - 1
      daik = adj2df(daik)
      daik = daik.loc[
        daik["source"] < daik["target"], "weight"
      ].to_numpy().ravel()
      # Filter dataframes ----
      dD = dD.loc[
        dD["source"] < dD["target"], "weight"
      ].to_numpy().ravel()
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "dist" : dD,# / np.max(dD["weight"]),
          "target similarity" : daki,
          "source similarity" : daik
        }
      )
      # Create figures ----
      fig, ax = plt.subplots(1, 3)
      sns.scatterplot(
        data=data, x="dist", y="target similarity",
        s=s, ax=ax[0]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["dist"], data["target similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["dist"], data["target similarity"]
      )
      ax[0].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[0].transAxes
      )
      sns.scatterplot(
        data=data, x="dist", y="source similarity",
        s=s, ax=ax[1]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["dist"], data["source similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["dist"], data["source similarity"]
      )
      ax[1].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[1].transAxes
      )
      sns.scatterplot(
        data=data,
        x="source similarity", y="target similarity",
        s=s, ax=ax[2]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["source similarity"],
        data["target similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["source similarity"], data["target similarity"]
      )
      ax[2].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[2].transAxes
      )
      fig.set_figwidth(15)
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
          plot_path, "similarity_plots_{}.png".format(self.linkage)
        ),
        dpi=300
      )
    else:
      print("No similarities and distance plots")
      
  def plot_aki(self, s=1, on=True):
    if on:
      print("Plot aki-aik!!!")
      daki = adj2df(self.aki.copy())
      daik = adj2df(self.aik.copy())
      # Filter dataframes ----
      daik = daik.loc[daik["source"] < daik["target"]]
      daki = daki.loc[daki["source"] < daki["target"]]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "target similarity" : daki["weight"],
          "source similarity" : daik["weight"]
        }
      )
      # Create figures ----
      fig, _ = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="source similarity", y="target similarity",
        s=s
      )
      fig.set_figwidth(5)
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
          plot_path, "sim_plot_{}.png".format(self.linkage)
        ),
        dpi=300
      )
    else:
      print("No aki-aik")
  
  def plot_histoscatter(self, A, partition, on=False, **kwargs):
    if on:
      print("Plot histoscatter for A!!!")
      # Get new partition
      new_partition = skim_partition(partition)
      _, fq = sort_by_size(new_partition, self.nodes)
      print(fq)
      new_partition_2 = np.zeros(len(new_partition), dtype=int)
      c = 0
      for k in fq.keys():
        if k != -1:
          new_partition_2[new_partition == k] = c
          c += 1
        else:
          new_partition_2[new_partition == k] = k
      ##
      dA = adj2df(A)
      dA = dA.loc[dA.weight != 0]
      dA.weight = np.log(dA.weight)
      dA["community"] = "off-diagonal"
      unique_communities = np.unique(new_partition_2)
      # Assign id to edges
      for com in unique_communities:
        if com != -1: name_com = f"cluster_{com}"
        else: name_com = "overlapping_nodes"
        nodes_com = np.where(new_partition_2 == com)[0]
        n_com = len(nodes_com)
        for i in np.arange(n_com):
          for j in np.arange(i+1, n_com):
            dA.community.loc[
              (dA.source == nodes_com[i]) & (dA.target == nodes_com[j])
            ] = name_com
            dA.community.loc[
              (dA.target == nodes_com[i]) & (dA.source == nodes_com[j])
            ] = name_com
      dA = dA.sort_values(by="community", ignore_index=True)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        hue="community",
        element="poly",
        stat="density",
        fill=False,
        common_norm=True,
        alpha=0.5,
        ax=ax
      )
      ax.set_yscale("log")
      sns.move_legend(ax, "upper left", bbox_to_anchor=(1.03, 1))
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
          plot_path, "histoscatter_fln.png"
        ),
        dpi=300
      )
    else: print("No histoscatter") 
