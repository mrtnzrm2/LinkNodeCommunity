# Standard libs ----
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os.path import join
# Personal libs ----
from plotting_modules.plotting_serial import PLOT_S

class PLOT_OS(PLOT_S):
  def __init__(self, hrh) -> None:
    super().__init__(hrh)
    self.data_overlap = hrh.data_overlap

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
