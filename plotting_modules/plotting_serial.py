import seaborn as sns
sns.set_theme()
import numpy as np
from os.path import join
from scipy import   stats
import pandas as pd
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
    self.node_entropy = hrh.node_entropy
    self.link_entropy = hrh.link_entropy
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
  
  def histogram_clustering_similarity(self, on=True, c=True, **kwargs):
    if on:
      print("Plot clustering similarity histogram!!!")
      subdata = self.data.copy()
      # Average score ----
      mean_results = subdata.groupby(["sim", "score"]).mean().reset_index()
      ## MAXMU
      mean_nmi = mean_results["values"].loc[(mean_results.sim == "NMI") & (mean_results.score == "_maxmu")].to_numpy()
      mean_nmi = np.round(mean_nmi, 3)[0]
      mean_omega = mean_results["values"].loc[(mean_results.sim == "OMEGA") & (mean_results.score == "_maxmu")].to_numpy()
      mean_omega = np.round(mean_omega, 3)[0]
      ## X
      mean_nmi_x = mean_results["values"].loc[(mean_results.sim == "NMI") & (mean_results.score == "_X")].to_numpy()
      mean_nmi_x = np.round(mean_nmi_x, 3)[0]
      mean_omega_x = mean_results["values"].loc[(mean_results.sim == "OMEGA") & (mean_results.score == "_X")].to_numpy()
      mean_omega_x = np.round(mean_omega_x, 3)[0]
      subdata["sim"] = subdata["sim"].map({"NMI" : f"NMI -> maxmu: {mean_nmi}     X: {mean_nmi_x}", "OMEGA" : f"OMEGA -> maxmu: {mean_omega}      X: {mean_omega_x}"})
      # print(mean_results)
      # Create figure ----
      if c:
        subdata["score"] = [s.replace("_", "") for s in subdata["score"]]
        subdata = subdata.loc[subdata.score != "D"] ####
        g = sns.FacetGrid(
          data=subdata,
          col="sim",
          hue="score",
          sharex=False, sharey=False,
          aspect=1.3, height=6
        )
        g.map_dataframe(
          sns.histplot,
          x="values",
          stat="count",
          alpha=0.4,
          common_norm=False,
          # bin_width=0.05,
          **kwargs
        )
        g.add_legend() 
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
          stat="count",
          **kwargs
        )
      # g.set_titles({"NMI" : r"$\mu_{NMI}: }$" + f"{mean_nmi:.3f}", "OMEGA" : r"$\mu_{OMEGA}$:" + f"{mean_omega:.3f}"})
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
          plot_path, "clustering_similarity_{}.png".format(
            self.linkage
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
      # ax[0, 0].text(
      #   x = 0.5, y = 0.5,
      #   s = "{:.5f}\n{:.5f}".format(
      #     os_ttest.pvalue, osd_ttest.pvalue
      #   ),
      #   ha='center', va='center',
      #   transform=ax[0, 0].transAxes
      # )
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

  def plot_measurements_Entropy(self, on=True):
    if on:
      print("Visualize Entropy iterations!!!")
      # Create data ----
      data = pd.concat([self.node_entropy, self.link_entropy])
      mx = data.iloc[
        data.groupby(["c", "dir", "data"])["S"].transform("idxmax").drop_duplicates(keep="first").to_numpy()
      ].sort_values("c", ascending=False)
      print(mx)
      # Create figure ----
      g = sns.FacetGrid(
        data=data,
        col = "c",
        hue = "dir",
        aspect=1.2,
        col_wrap=2,
        sharex=False,
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="level",
        y="S",
        style="data",
        alpha=0.4
      )#.set(xscale="log")
      g.add_legend()
      # Arrange path ----
      plot_path = join(self.plot_path, "Features")
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

  def plot_measurements_Entropy_noodle(self, on=True):
    if on:
      print("Visualize Entropy iterations!!!")
      # Create data ----
      data = pd.concat([self.node_entropy, self.link_entropy])
      mx = data.iloc[
        data.groupby(["c", "dir", "data"])["S"].transform("idxmax").drop_duplicates(keep="first").to_numpy()
      ].sort_values("c", ascending=False)
      print(mx)
      _cmap_v = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True)
      _cmap_h = sns.cubehelix_palette(as_cmap=True)
      color_palette = {col : _cmap_v(-col / data.iter.max()) for col in np.unique(data.loc[(data.data == "0") & (data.dir == "V")].iter)}
      color_palette.update({col : _cmap_h(col / data.iter.max()) for col in np.unique(data.loc[(data.data == "0") & (data.dir == "H")].iter)})
      data.iter.loc[(data.data == "1" )] = 0.1
      data["width"] = "thick"
      data.width.loc[data.data == "0"] = "thin"
      color_palette.update({0.1 : "#dc4d01"})
      # Create figure ----
      g = sns.FacetGrid(
        data=data,
        col = "c",
        row = "dir",
        hue = "iter",
        aspect=1.2,
        palette=color_palette,
        sharex=False,
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="level",
        y="S",
        size="width",
        sizes={"thick" : 1, "thin" : 0.5},
        estimator=None,
        alpha=0.4
      )#.set(xscale="log")
      # g.add_legend()
      plt.legend([],[], frameon=False)
      # Arrange path ----
      plot_path = join(self.plot_path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        join(
          plot_path, "Entropy_levels_noodle.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No Entropy iterations")

  def plot_entropy(self, s=3, on=True):
    if on:
      # Modified after gathering new data
      print("Plot Entropy Sh, Sv!!!")
      fig, ax = plt.subplots(1, 1)
      sns.scatterplot(
        data=self.entropy,
        x = "Sh",
        y = "Sv",
        s = s,
        ax = ax
      )
      sns.scatterplot(
        data=self.entropy[self.entropy.data == "1"],
        x = "Sh",
        y = "Sv",
        color = "#C70039",
        s = s,
        ax = ax
      )
      ax.set_xlabel("SV")
      ax.set_ylabel("SH")
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
        join(plot_path, "entropy.png"),
        dpi=300
      )
      plt.close()
    else: print("No Entropy Sh, Sv")
  
  def plot_entropy_histplot(self, on=True):
    if on:
      print("Plot entropy histograms")
      # Get data ----
      fig, ax = plt.subplots(1, 2, figsize=(9, 4))
      h1 = sns.histplot(
        data=self.entropy.loc[self.entropy.data == "0"],
        x="Sh",
        stat="count",
        color=sns.color_palette("deep", n_colors=2)[1],
        ax=ax[0]
      )
      h1.axvline(self.entropy.Sh.loc[self.entropy.data == "1"].to_numpy())
      h2 = sns.histplot(
        data=self.entropy.loc[self.entropy.data == "0"],
        x="Sv",
        stat="count",
        color=sns.color_palette("deep", n_colors=2)[1],
        ax=ax[1]
      )
      h2.axvline(self.entropy.Sv.loc[self.entropy.data == "1"].to_numpy())
      ax[0].set_xlabel("Sv")
      ax[1].set_xlabel("Sh")
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
        join(plot_path, "entropy_hist.png"),
        dpi=300
      )
      plt.close()
    else: print("No entropy histograms")



