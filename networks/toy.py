# Standard libs ----
from pathlib import Path
from os.path import join
import numpy as np
# Personalized libs ----
from networks.base import BASE

class TOY(BASE):
  def __init__(
    self, A, linkage, nlog10=False, lookup=False, cut=False, 
    **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.A = A.copy()
    self.nodes, self.rows = A.shape
    # Set ANALYSIS NAME ----
    self.analysis = linkage.upper() + "_{}_{}".format(
      self.rows, self.nodes
    )
    if nlog10:
      self.analysis = self.analysis + "_l10"
    if lookup:
      self.analysis = self.analysis + "_lup"
    if cut:
      self.analysis = self.analysis + "_cut"
    if "EC" in kwargs.keys():
      if kwargs["EC"]:
       self.analysis = self.analysis + "_EC" 
    # Create paths ----
    if "plot_path" in kwargs.keys():
      self.plot_path = kwargs["plot_path"]
    else:
      self.plot_path = join(
        "../plots", "TOY", self.common_path,
        self.analysis, self.mode, self.topology
      )
    if "pickle_path" in kwargs.keys():
      self.pickle_path = kwargs["pickle_path"]
    else:
      self.pickle_path = join(
        "../pickle", "TOY", self.common_path,
        self.analysis, self.mode, self.topology
      )
    # Overlap ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)
    # NOCS dict ----
    self.data_nocs = {}

  def create_plot_directory(self):
    Path(
      self.plot_path
    ).mkdir(exist_ok=True, parents=True)
    
  def create_pickle_directory(self):
    Path(
      self.pickle_path
    ).mkdir(exist_ok=True, parents=True)

  def set_labels(self, labels):
    self.labels = labels
    self.struct_labels = labels