# Standard libs ----
from pathlib import Path
import numpy as np
import pandas as pd
from os.path import join
# Personal libs ----
from various.network_tools import *
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion

class TOYH:
  def __init__(self) -> None:
    # Entropy
    self.node_entropy = pd.DataFrame()
    self.link_entropy = pd.DataFrame()

  def set_subfolder(self, subfolder):
    self.subfolder = subfolder

  def set_pickle_path(self, H : Hierarchy, bias=0):
     # Get pickle location ----
    pickle_path = H.pickle_path.split("/")
    self.pickle_path = ""
    for i in np.arange(len(pickle_path)):
      self.pickle_path = join(
        self.pickle_path, pickle_path[i]
      )
      if pickle_path[i] == H.analysis: break
    self.pickle_path = join(
      self.pickle_path, H.mode, H.subfolder
    )
    self.pickle_path = join(self.pickle_path, f"b_{bias}")
    Path(self.pickle_path).mkdir(exist_ok=True, parents=True)

  def set_iter(self, it):
    self.iter = it
    
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