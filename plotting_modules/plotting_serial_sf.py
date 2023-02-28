import seaborn as sns
sns.set_theme()
import numpy as np
from os.path import join
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

class PLOT_S_SF:
  def __init__(self, hrh) -> None:
    # Get attributes from hrh ----
    self.data = hrh.data
    self.stats = hrh.stats
    self.measures = hrh.data_measures
    self.plot_path = hrh.plot_path
    self.linkage = hrh.linkage

  def plot_measurements_D_noodle(self, on=False, **kwargs):
    if on:
      print("Plot D noodle iterations")
      data = self.measures[["K", "D", "iter"]].copy()
      data.iter = data.iter.to_numpy().astype(int)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="D",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
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
      data = self.measures[["K", "X", "iter"]].copy()
      data.iter = data.iter.to_numpy().astype(int)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="X",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
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

  def plot_measurements_ntrees_noodle(self, on=False, **kwargs):
    if on:
      print("Plot D noodle iterations")
      data = self.measures[["K", "ntrees", "iter"]].copy()
      data.iter = data.iter.to_numpy().astype(int)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="ntrees",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
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

  def plot_measurements_mu_noodle(self, on=False, **kwargs):
    if on:
      print("Plot mu noodle iterations")
      data = self.measures[["K", "mu", "iter"]].copy()
      data.iter = data.iter.to_numpy().astype(int)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="mu",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
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

  def plot_measurements_ordp_noodle(self, on=False, **kwargs):
    if on:
      print("Plot order parameter noodle iterations")
      data = self.measures[["K", "m", "iter"]].copy()
      data.iter = data.iter.to_numpy().astype(int)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="m",
        hue="iter",
        alpha=0.4,
        lw=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        estimator=None,
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
          plot_path, "ordp_logK_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No order parameter noodle iterations")

  def plot_measurements_D(self, on=False, **kwargs):
    if on:
      print("Plot D iterations")
      data = self.measures[["K", "D", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="D",
        alpha=0.7,
        errorbar="sd",
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
      data = self.measures[["K", "X", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="X",
        alpha=0.7,
        errorbar="sd",
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
      data = self.measures[["K", "mu", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="mu",
        alpha=0.7,
        errorbar="sd",
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
      data = self.measures[["K", "ntrees", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="ntrees",
        errorbar="sd",
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
      data = self.measures[["K", "m", "iter"]]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="K",
        y="m",
        errorbar="sd",
        alpha=0.7,
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

  def histogram_clustering_similarity(self, c=True, on=True, **kwargs):
    if on:
      print("Plot serial clustering similarity histogram!!!")
      data = self.data.copy()
      print(data)
      # Zero NMI ----
      data["values"].loc[np.isnan(data.sim == "NMI")] = 0
      data["values"].loc[np.isnan(data.sim == "OMEGA")] = 0
      data.c = [s.replace("_", "") for s in data.c]
      if c:
        g = sns.FacetGrid(
          data=data,
          col="sim",
          hue="c",
          sharex=False, sharey=False,
          aspect=1.3, height=6
        )
        g.map_dataframe(
          sns.histplot,
          x="values",
          stat="probability",
          alpha=0.6,
          **kwargs
        )
        g.add_legend()
      else:
        g = sns.FacetGrid(
          data=data,
          col="sim",
          sharex=False, sharey=False,
          aspect=1.3, height=6
        )
        g.map_dataframe(
          sns.histplot,
          x="values",
          stat="probability",
          **kwargs
        )
      # plt.subplots_adjust(bottom=0.32)
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
          plot_path, "adj_hist_{}.png".format(
            self.linkage
          )
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No serial clustering similarity Histogram")
  
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
        stat = "density",
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
        stat = "density",
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
        stat = "density",
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
        stat = "density",
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
        stat = "density",
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
        stat = "density",
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



