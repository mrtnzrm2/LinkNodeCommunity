# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Any
import plotly.express as px
import os
from collections import Counter
# Personal libs ----
from plotting_modules.plotting_H import Plot_H
from modules.sign.hierarmerge import Hierarchy as signHierarchy
from modules.hierarmerge import Hierarchy
from various.network_tools import *
from modules.flatmap import FLATMAP

class Plot_HCP(Plot_H):
  def __init__(self, NET, H) -> None:
    super().__init__(NET, H)
    self.subject_id = NET.subject_id


  def plotly_nodetimeseries_Z(self, R : int, partition : list, small_set=None, on=True):
    if on:
      print("Plot nodetimesries Z!!!")
      if isinstance(self.subject_id, str):
        nodetimeseries_matrix = pd.read_table(f"../CSV/HCP/nodetimeseries_{self.nodes}/{self.subject_id}.txt", sep=" ").to_numpy().T
      else:
        nodetimeseries_path = f"../CSV/HCP/nodetimeseries_{self.nodes}"
        list_nodetimeseries = os.listdir(nodetimeseries_path)

        nodetimeseries_matrix = []
        for i, file in enumerate(list_nodetimeseries):
          try:
            nodetimeseries_matrix.append(pd.read_table(f"../CSV/HCP/nodetimeseries_{self.nodes}/{file}", sep=" ").to_numpy())
          except:
            print(file)
        nodetimeseries_matrix = np.array(nodetimeseries_matrix)
        nodetimeseries_matrix = np.nanmean(nodetimeseries_matrix, axis=0).T

      t1 = 100
      t2 = 1199
      nodetimeseries_matrix = nodetimeseries_matrix[:, t1:t2]
      T = nodetimeseries_matrix.shape[1]

      if small_set:
        c = dict(Counter(partition))
        c = dict(sorted(c.items(), key=lambda item: item[1]))
        c = {k : v for k, v in c.items() if k != -1}
        kc = list(c.keys())[0]

      clusters = np.unique(partition)
      clusters = np.array([i for i in clusters if i != -1])

      data = pd.DataFrame()

      for cls in clusters:
        if small_set:
          if cls != kc: continue

        nodes = np.where(partition == cls)[0]
        for n in nodes:
          subdata = pd.DataFrame(
                {
                  "clusters" : [cls] * T,
                  "nodetimeserie" : [n] * T,
                  "average signal" : nodetimeseries_matrix[n, :].ravel(),
                  "frame" : np.arange(t1, t2)
                }
              )
          subdata["smooth signal"] = subdata["average signal"].rolling(7).mean()
          data = pd.concat([data, subdata], ignore_index=T)

      if not small_set:
        fig = px.line(
          data,
          x="frame", y="smooth signal",
          color="nodetimeserie",
          facet_row="clusters",
          template="plotly_dark",
          height=1500
        )
      else:
        fig = px.line(
          data,
          x="frame", y="smooth signal",
          color="nodetimeserie",
          facet_row="clusters",
          template="plotly_dark"
        )

      #  Arrange path ----
      fname = "nodetimeseries"
      if small_set:
        fname = "nodetimeseries_small"
      plot_path = os.path.join(
        self.path, fname
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      if isinstance(self.subject_id, str):
        fig.write_html(f"{plot_path}/{R}_{self.subject_id}.html")
      else:
        fig.write_html(f"{plot_path}/{R}.html")
    else:
      print("No nodetimeseries Z.")
    