# Standard libs ----
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join
# Personal libs ----
from various.network_tools import AD_NMI_label, get_H_from_BH_with_maxmu
from modules.hierarmerge import Hierarchy

class SCALEHRH:
  def __init__(self, linkage) -> None:
    self.linkage = linkage
    self.data = pd.DataFrame()
    self.stats = pd.DataFrame()
    self.data_measures = pd.DataFrame()
    self.node_entropy = pd.DataFrame()
    self.link_entropy = pd.DataFrame()

  def update_entropy(self, s):
    dim = s[0].shape[1]
    self.node_entropy = pd.concat(
      [
        self.node_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[0].ravel(), s[1].ravel()]),
            "c" : ["node_hierarchy"] * 2 * dim  + ["node_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4),
            "iter" : [(self.iter+1)] * dim + [-(self.iter+1)] * dim + [(self.iter+1)] * dim + [-(self.iter+1)] * dim
          } 
        )
      ], ignore_index=True
    )
    dim = s[2].shape[1]
    self.link_entropy = pd.concat(
      [
        self.link_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[2].ravel(), s[3].ravel()]),
            "c" : ["link_hierarchy"] * 2 * dim  + ["link_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4),
            "iter" : [(self.iter+1)] * dim + [-(self.iter+1)] * dim + [(self.iter+1)] * dim + [-(self.iter+1)] * dim
          }
        )
      ], ignore_index=True
    )

  def set_data_measurements(self, HH : Hierarchy, iter):
    H = get_H_from_BH_with_maxmu(HH)[
      ["K", "mu", "X", "D", "m", "ntrees"]
    ]
    H["iter"] = [str(iter)] * H.shape[0]
    self.data_measures = pd.concat(
      [self.data_measures, H],
      ignore_index=True
    )

  def set_subfolder(self, subfolder):
    self.subfolder = subfolder

  def set_nmi_nc(self, l1, l2, **kwargs):
    #create subdata ----
    if "score" in kwargs.keys():
      subdata = pd.DataFrame(
        {
          "NMI" : [AD_NMI_label(l1, l2)],
          "c" : [kwargs["score"]],
          "iter" : [self.iter]
        }
      )
    else:
      subdata = pd.DataFrame(
        {
          "NMI" : [AD_NMI_label(l1, l2)],
          "c" : ["node community"],
          "iter" : [self.iter]
        }
      )
    # Merge with data ----
    self.data = pd.concat(
      [self.data, subdata], ignore_index=True
    )

  def set_iter(self, iter):
    self.iter = iter

  def set_pickle_path(self, H : Hierarchy):
     # Get pickle location ----
    pickle_path = H.pickle_path.split("/")
    self.pickle_path = ""
    for i in np.arange(len(pickle_path)):
      self.pickle_path = join(
        self.pickle_path, pickle_path[i]
      )
      if pickle_path[i] == self.subfolder: break
    self.pickle_path = join(
      self.pickle_path, H.mode, H.subfolder
    )
    Path(self.pickle_path).mkdir(exist_ok=True, parents=True)

  def set_plot_path(self, H : Hierarchy):
    plot_path = H.plot_path.split("/")
    self.plot_path = ""
    for i in np.arange(len(plot_path)):
      self.plot_path = join(
        self.plot_path, plot_path[i]
      )
      if plot_path[i] == self.subfolder: break
    self.plot_path = join(
      self.plot_path, H.mode, H.subfolder
    )
    Path(self.plot_path).mkdir(exist_ok=True, parents=True)
