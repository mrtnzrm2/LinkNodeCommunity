# Standard libs ----
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os.path import join
# Personal libs ----
from various.network_tools import *
from plotting_modules.plotting_serial import PLOT_S

class PLOT_OS(PLOT_S):
  def __init__(self, hrh) -> None:
    super().__init__(hrh)
    self.data_overlap = hrh.data_overlap
    self.association_one = hrh.association_one
    self.Z = hrh.Z
    self.kr = hrh.kr
    self.nodes = hrh.nodes
    self.association_zero = hrh.association_zero
    self.labels = hrh.labels

  def association_heatmap(self, score, on=True, **kwargs):
    if on:
      print(f"Plot association {score} matrix")
      from scipy.cluster.hierarchy import dendrogram, cut_tree
      # Getting the r of the score ----
      r = [a for a in self.kr.R.loc[(self.kr.score == score) & (self.kr.data == "1")]]
      r = r[0]
      # Getting the dendrogram order of Z ----
      one_order = np.array(dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      # Preparing the labels of Z at r -----
      one_rlabel = cut_tree(self.Z, n_clusters=r).reshape(-1)
      one_rlabel = skim_partition(one_rlabel)[one_order]
      C = [i+1 for i in np.arange(len(one_rlabel)-1) if one_rlabel[i] != one_rlabel[i+1]]
      D = np.where(one_rlabel == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      # Preparing matrix ----
      zero_matrix = self.association_zero[score][one_order,:][:, one_order]
      zero_matrix[zero_matrix == 0] = np.nan
      one_labels = self.labels[one_order]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.heatmap(
        zero_matrix,
        xticklabels=one_labels,
        yticklabels=one_labels,
        cmap=sns.color_palette("mako", as_cmap=True),
        ax=ax
      )
      # Add lines denoting communities ----
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      # Configure figure ----
      fig.set_figwidth(11.5)
      fig.set_figheight(9.5)
      # Arrange path ----
      plot_path = join(self.plot_path, f"Features")
       # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(plot_path, f"association_{score}.png"), dpi=300
      )
      plt.close()
    else: print(f"No {score} association matrix")


  def histogram_overlap_scores(self, on=True, **kwargs):
    if on:
      print("Plot overlap scores histogram!!!")
      print(
        "Mean acc1: {:7f}\nMean acc2: {:.7f}".format(
          self.data_overlap["ACC1"].mean(),
          self.data_overlap["ACC2"].mean()
        )
      )
      # Create figure ----
      if "c" in kwargs.keys():
        if kwargs["c"]:
          self.data_overlap["c"] = [s.replace("_", "") for s in self.data_overlap["c"]]
          fig, ax = plt.subplots(1,2)
          sns.histplot(
            data=self.data_overlap,
            x = "ACC1",
            ax = ax[0],
            hue = "c"
          )
        else:
          fig, ax = plt.subplots(1,2)
          sns.histplot(
            data=self.data_overlap,
            x = "ACC1",
            ax = ax[0]
          )
      else:
        fig, ax = plt.subplots(1,2)
        sns.histplot(
          data=self.data_overlap,
          x = "ACC1",
          ax = ax[0]
        )
      ax[0].text(
        0.5, 1.05,
        "Average: {:.4f}   Std: {:.4f}".format(
          self.data_overlap["ACC1"].mean(),
          self.data_overlap["ACC1"].std()
        ),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[0].transAxes
      )
      ax[0].set(xlabel="tp/total")
      # Create figure ACC2 ----
      if "c" in kwargs.keys():
        if kwargs["c"]:
          sns.histplot(
            data=self.data_overlap,
            x = "ACC2",
            ax = ax[1],
            hue = "c"
          )
        else:
          sns.histplot(
            data=self.data_overlap,
            x = "ACC2",
            ax = ax[1]
          )
      else:
        sns.histplot(
          data=self.data_overlap,
          x = "ACC2",
          ax = ax[1]
        )
      ax[1].text(
        0.5, 1.05,
        "Average: {:.4f}    Std: {:.4f}".format(
          self.data_overlap["ACC2"].mean(),
          self.data_overlap["ACC2"].std()
        ),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[1].transAxes
      )
      ax[1].set(xlabel="fp/pred")
      # figure custom ---
      fig.set_figwidth(12)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Name ----
      name = ""
      if "name" in kwargs.keys():
        name = kwargs["name"]
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "overlap_scores_{}{}.png".format(
            self.linkage, name
          )
        ),
        dpi=300
      )

  def histogram_overlap(self, score, on=True, **kwargs):
    if on:
      print("Histogram overlap frequency!!!")
      subdata = self.data_overlap.loc[
        self.data_overlap.score == score
      ]
      fig, ax = plt.subplots(1,1)
      sns.histplot(
        data = subdata,
        x = "Areas",
        hue = "data",
        stat = "count",
        common_norm=False,
        ax = ax
      )
      # Rotate axis ----
      plt.xticks(rotation=90)
      # Get areas from data ----
      data_areas = subdata.loc[
        subdata.data == "1",
        "Areas"
       ].to_numpy()
      for i in np.arange(len(data_areas)):
        ax.axvline(data_areas[i], color="red")
      fig.set_figheight(9)
      fig.set_figwidth(15)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "overlap_freq_{}_{}.png".format(
            self.linkage, score
          )
        ),
        dpi=300
      )
      plt.close()
