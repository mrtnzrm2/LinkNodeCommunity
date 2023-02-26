# Standard libs ----
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from pathlib import Path
from os.path import join
# Personal libs ----
from plotting_modules.plotting_serial_sf import PLOT_S_SF

class PLOT_OS_SF(PLOT_S_SF):
  def __init__(self, hrh) -> None:
    super().__init__(hrh)
    self.data_overlap = hrh.data_overlap

  def histogram_overlap_scores(self, c=True, on=True, **kwargs):
    if on:
      print("Plot overlap scores histogram!!!")
      print("Mean score:")
      data = self.data_overlap.copy()
      print(self.data_overlap.groupby("c").mean())
      # Create figure ----
      fig, ax = plt.subplots(1, 3)
      if c:
        data["c"] = [s.replace("_", "") for s in data["c"]]
        sns.histplot(
          data=data,
          x = "sensitivity",
          ax = ax[0],
          hue = "c",
          **kwargs
        )
        sns.histplot(
          data=data,
          x = "specificity",
          ax = ax[1],
          hue = "c",
          **kwargs
        )
        sns.histplot(
          data=data,
          x = "omega",
          ax = ax[2],
          hue = "c",
          **kwargs
        )
      else:
        sns.histplot(
          data=data,
          x = "sensitivity",
          ax = ax[0]
        )
        sns.histplot(
          data=data,
          x = "specificity",
          ax = ax[1]
        )
        sns.histplot(
          data=data,
          x = "omega",
          ax = ax[2]
        )
      # figure custom ---
      fig.set_figwidth(16)
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
          plot_path, "overlap_scores_{}.png".format(
            self.linkage
          )
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No overlap scores histogram")

  def ROC_OCN(self, on=False, **kwargs):
    if on:
      print("ROC_OCN")
      data = self.data_overlap.copy()
      data["FPR"] = 1 - data.specificity
      data = data.sort_values(by="FPR", ignore_index=True)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      if "c" in data.columns:
        data["c"] = [s.replace("_", "") for s in data["c"]]
        scores = np.unique(data.c)
        cmap = sns.color_palette("deep", len(scores))
        for i, score in enumerate(scores):
          sns.regplot(
            data=data.loc[data["c"] == score],
            x="FPR",
            y="sensitivity",
            color=cmap[i],
            lowess=True,
            scatter=False,
            ax=ax
          )
        sns.scatterplot(
          data=data,
          x="FPR",
          y="sensitivity",
          hue="c",
          s=15,
          palette=cmap,
          ax=ax,
          **kwargs
        )
      else:
        sns.scatterplot(
          data=data,
          x="FPR",
          y="sensitivity",
          s=15,
          ax=ax
        )
      sns.lineplot(
        x=np.linspace(0, 1, 10),
        y=np.linspace(0, 1, 10),
        color="black",
        linestyle="dashed",
        lw=1,
        ax=ax
      )
      ax.set_ylabel("TPR")
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
          plot_path, "ROC_OCN.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No ROC_OCN")

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
        stat = "probability",
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
    else:
      print("No hist overlap frequency")
