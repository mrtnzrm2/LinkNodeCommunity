import seaborn as sns
sns.set_theme()
import numpy as np
from os.path import join
from scipy import   stats
import matplotlib.pyplot as plt
from pathlib import Path
from networks_serial.hrh import HRH

class PLOT_S:
  def __init__(self, hrh : HRH) -> None:
    # Get attributes from hrh ----
    self.data = hrh.data
    self.stats = hrh.stats
    self.NH = hrh.data_homoegeity
    self.measures = hrh.data_measures
    self.kr = hrh.kr
    self.plot_path = hrh.plot_path
    self.linkage = hrh.linkage
    self.labels = hrh.labels

  def scatterplot_NH(self, on=True, **kwargs):
    if on:
      print("Plot TNH scatterplot!!!")
      nh_data = self.NH.TNH.loc[self.NH.data == "1"]
      order = np.argsort(nh_data)
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))
      sns.stripplot(
        data=self.NH,
        x= "area",
        y="TNH",
        hue="data",
        s=3,
        order=self.labels[order],
        # aspect=2,
        ax=ax
      )
      plt.xticks(rotation=90)
      fig.tight_layout()
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
          plot_path, "TNH_{}{}.png".format(
            self.linkage, name
          )
        ),
        dpi=300
      )
    else: print("No TNH")
  
  def histogram_clustering_similarity(self, score, on=True, **kwargs):
    if on:
      print("Plot clustering similarity histogram!!!")
      subdata = self.data.loc[self.data.score == score]
      # Average score ----
      print(subdata.groupby("sim").mean().reset_index())
      # Create figure ----
      if "c" in kwargs.keys():
        if kwargs["c"]:
          subdata["c"] = [s.replace("_", "") for s in subdata["c"]]
          g = sns.FacetGrid(
            data=subdata,
            col="sim",
            sharex=False, sharey=False,
            aspect=1.3, height=6
          )
          g.map_dataframe(
            sns.histplot,
            x="values",
            stat="probability",
            hue="c"
          )
          g.legend()
      else:
        g = sns.FacetGrid(
          data=subdata,
          col="sim",
          sharex=False, sharey=False,
          aspect=1.3, height=6
        )
        g.map_dataframe(
          sns.histplot,
          x="values",
          stat="probability"
        )
      # ax.text(
      #   0.5, 1.05,
      #   r"$\mu$" + f": {subdata.NMI.mean():.4f}    " + r"$\sigma$" + f": {subdata.NMI.std():.4f}",
      #   horizontalalignment='center',
      #   verticalalignment='center',
      #   transform=ax.transAxes
      # )
      # plt.xlabel("ADJ_NMI")
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
          plot_path, "clustering_similarity_{}_{}.png".format(
            self.linkage, score
          )
        ),
        dpi=300
      )
    else:
      print("No clustering similarity histogram")
  
  def plot_stats(self, alternative="greater", on=True, **kwargs):
    if on:
      print("Plot stats!!!")
      # Create figures ----
      fig, ax = plt.subplots(2, 3)
      ## Compute p-value ----
      subdata = self.stats.loc[
          (self.stats["X"] == "source") &
          (self.stats["Y"] == "target")
        ]
      mu_0 = subdata["Cor"].loc[
        subdata["data"] == "1"
      ].to_numpy()
      x = subdata["Cor"].loc[
        subdata["data"] == "0"
      ]
      ## One-sample t-test ----
      os_ttest = stats.ttest_1samp(
        x, popmean=mu_0
      )
      ## One-sided one-sample t-test ----
      osd_ttest = stats.ttest_1samp(
        x, popmean=mu_0,
        alternative=alternative
      )
      sns.histplot(
        data=subdata,
        x = "Cor",
        hue = "data",
        stat = "probability",
        ax = ax[0, 0]
      )
      ax[0, 0].text(
        x = 0.5, y = 1.05,
        s = "Source & target similarity",
        ha='center', va='bottom',
        transform=ax[0, 0].transAxes
      )
      ax[0, 0].text(
        x = 0.5, y = 0.5,
        s = "{:.5f}\n{:.5f}".format(
          os_ttest.pvalue[0], osd_ttest.pvalue[0]
        ),
        ha='center', va='center',
        transform=ax[0, 0].transAxes
      )
      # Vertical line ----
      ax[0, 0].axvline(mu_0, color="red")
      # Get R-squared ----
      num = self.stats.loc[
          (self.stats["X"] == "source") &
          (self.stats["Y"] == "target") &
          (self.stats["data"] == "1"),
          "R-squared"
        ].to_numpy()
      sns.histplot(
        data=self.stats[
          (self.stats["X"] == "source") &
          (self.stats["Y"] == "target")
        ],
        x = "R-squared",
        hue = "data",
        stat = "probability",
        ax = ax[1, 0]
      )
      # Vertical line ----
      ax[1, 0].axvline(num, color="red")
      # Get Cor ----
      num = self.stats.loc[
        (self.stats["X"] == "distance") &
        (self.stats["Y"] == "target") &
        (self.stats["data"] == "1"),
        "Cor"
      ].to_numpy()
      sns.histplot(
        data=self.stats.loc[
          (self.stats["X"] == "distance") &
          (self.stats["Y"] == "target")
        ],
        x = "Cor",
        hue = "data",
        stat = "probability",
        ax = ax[0, 1]
      )
      ax[0, 1].text(
        x = 0.5, y = 1.05,
        s = "Distance & target similarity",
        ha='center', va='bottom',
        transform=ax[0, 1].transAxes
      )
      # Vertical line ----
      ax[0, 1].axvline(num, color="red")
      # Get R-squared ----
      num = self.stats.loc[
        (self.stats["X"] == "distance") &
        (self.stats["Y"] == "target") &
        (self.stats["data"] == "1"),
        "R-squared"
      ].to_numpy()
      sns.histplot(
        data=self.stats[
          (self.stats["X"] == "distance") &
          (self.stats["Y"] == "target")
        ],
        x = "R-squared",
        hue = "data",
        stat = "probability",
        ax = ax[1, 1]
      )
      # Vertical line ----
      ax[1, 1].axvline(num, color="red")
      # Get Cor ----
      num = self.stats.loc[
        (self.stats["X"] == "distance") &
        (self.stats["Y"] == "source") &
        (self.stats["data"] == "1"),
        "Cor"
      ].to_numpy()
      sns.histplot(
        data=self.stats.loc[
          (self.stats["X"] == "distance") &
          (self.stats["Y"] == "source")
        ],
        x = "Cor",
        hue = "data",
        stat = "probability",
        ax = ax[0, 2]
      )
      ax[0, 2].text(
        x = 0.5, y = 1.05,
        s = "Distance & source similarity",
        ha='center', va='bottom',
        transform=ax[0, 2].transAxes
      )
      # Vertical line ----
      ax[0, 2].axvline(num, color="red")
      # Get R-squared ----
      num = self.stats.loc[
        (self.stats["X"] == "distance") &
        (self.stats["Y"] == "source") &
        (self.stats["data"] == "1"),
        "R-squared"
      ].to_numpy()
      sns.histplot(
        data=self.stats[
          (self.stats["X"] == "distance") &
          (self.stats["Y"] == "source")
        ],
        x = "R-squared",
        hue = "data",
        stat = "probability",
        ax = ax[1, 2]
      )
      # Vertical line ----
      ax[1, 2].axvline(num, color="red")
      fig.set_figheight(8)
      fig.set_figwidth(15)
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
          plot_path, "stats_{}{}.png".format(
            self.linkage, name
          )
        ),
        dpi=300
      )
      plt.close()
    else: print("No stats")

  def plot_measurements_D(self, on=False, **kwargs):
    if on:
      print("Plot D iterations")
      data = self.measures[["K", "D", "data", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="D",
        errorbar="sd",
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="D",
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "D_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No D iterations")

  def plot_measurements_X(self, on=False, **kwargs):
    if on:
      print("Plot X iterations")
      data = self.measures[["K", "X", "data", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="X",
        errorbar="sd",
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="X",
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "X_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No X iterations")

  def plot_measurements_mu(self, on=False, **kwargs):
    if on:
      print("Plot mu iterations")
      data = self.measures[["K", "mu", "data", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="mu",
        errorbar="sd",
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="mu",
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "mu_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No mu iterations")

  def plot_measurements_ntrees(self, on=False, **kwargs):
    if on:
      print("Plot ntrees iterations")
      data = self.measures[["K", "ntrees", "data", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="ntrees",
        errorbar="sd",
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="ntrees",
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "ntrees_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No ntrees iterations")

  def plot_measurements_ordp(self, on=False, **kwargs):
    if on:
      print("Plot order parameter iterations")
      data = self.measures[["K", "m", "data", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="m",
        errorbar="sd",
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="m",
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "ordp_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No order parameter iterations")

  def histogram_krs(self, score="", on=False):
    if on:
      print("Plotting K and R iterations")
      data = self.kr.copy()
      if score != "":
        data = data.loc[data.score == score]
      # Stats ----
      mean_k = data["K"].loc[data.data != "1"].mean(skipna=True)
      mean_r = data["R"].loc[data.data != "1"].mean(skipna=True)
      std_k = data["K"].loc[data.data != "1"].std(skipna=True)
      std_r = data["R"].loc[data.data != "1"].std(skipna=True)
      # ----
      k1 = data["K"].loc[data.data == "1"].iloc[0]
      r1 = data["R"].loc[data.data == "1"].iloc[0]
      # Create figure ----
      fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
      ## k ----
      sns.histplot(
        data=data,
        x="K",
        hue="data",
        ax=ax[0]
      )
      ax[0].axvline(k1, color="r")
      ax[0].text(
        x = 0.5, y = 1.05,
        s = r"$\mu$" + f": {mean_k:.2f}     " + r"$\sigma$" + f": {std_k:3f}",
        ha='center', va='bottom',
        transform=ax[0].transAxes
      )
      ## r ----
      sns.histplot(
        data=data,
        x="R",
        hue="data",
        ax=ax[1]
      )
      ax[1].axvline(r1, color="r")
      ax[1].text(
        x = 0.5, y = 1.05,
        s = r"$\mu$" + f": {mean_r:.2f} + " + r"$\sigma$" + f": {std_r:3f}",
        ha='center', va='bottom',
        transform=ax[1].transAxes
      )
      fig.tight_layout()
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
          plot_path, f"histo_kr{score}.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No KR iteration")

  def plot_measurements_D_noodle(self, on=False, **kwargs):
    if on:
      print("Plot D noodle iterations")
      data = self.measures[["K", "D", "data", "iter"]]
      data.iter.loc[data.data == "0"] = [int(i) for i in data.iter.loc[data.data == "0"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="D",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="D",
        color="#C70039",
        lw=1,
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "D_logK_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No D noodle iterations")

  def plot_measurements_X_noodle(self, on=False, **kwargs):
    if on:
      print("Plot X noodle iterations")
      data = self.measures[["K", "X", "data", "iter"]]
      data.iter.loc[data.data == "0"] = [int(i) for i in data.iter.loc[data.data == "0"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="X",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="X",
        lw=1,
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "X_logK_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No X noodle iterations")

  def plot_measurements_mu_noodle(self, on=False, **kwargs):
    if on:
      print("Plot mu noodle iterations")
      data = self.measures[["K", "mu", "data", "iter"]]
      data.iter.loc[data.data == "0"] = [int(i) for i in data.iter.loc[data.data == "0"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="mu",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="mu",
        lw=1,
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "mu_logK_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No mu noodle iterations")

  def plot_measurements_ntrees_noodle(self, on=False, **kwargs):
    if on:
      print("Plot ntrees noodle iterations")
      data = self.measures[["K", "ntrees", "data", "iter"]]
      data.iter.loc[data.data == "0"] = [int(i) for i in data.iter.loc[data.data == "0"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="ntrees",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="ntrees",
        lw=1,
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "ntrees_logK_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No ntrees noodle iterations")

  def plot_measurements_ordp_noodle(self, on=False, **kwargs):
    if on:
      print("Plot order parameter noodle iterations")
      data = self.measures[["K", "m", "data", "iter"]]
      data.iter.loc[data.data == "0"] = [int(i) for i in data.iter.loc[data.data == "0"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data.loc[data.data == "0"],
        x="K",
        y="m",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
        ax=ax
      )
      sns.lineplot(
        data=data.loc[data.data == "1"],
        x="K",
        y="m",
        lw=1,
        color="#C70039",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
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
          plot_path, "ordp_log_noodleK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No order parameter noodle iterations")



