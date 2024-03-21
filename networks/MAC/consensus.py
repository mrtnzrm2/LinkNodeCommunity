from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *

class base:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    self.subject = "MAC"
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "FLN"
    if "model" in kwargs.keys():
      self.model = kwargs["model"]
    else: self.model = ""
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    if "inj" in kwargs.keys():
      self.inj = kwargs["inj"]
    else: self.inj = ""
    if "nature" in kwargs.keys():
      self.nature = kwargs["nature"]
    else: self.nature = ""
    if "version" in kwargs.keys():
      self.version = kwargs["version"]
    else: self.version = ""
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature,
      self.model, self.inj, "Consensus"
    )

class CONSENSUS(base):
  def __init__(
    self, max_iter, nodes, rows, linkage : str, mode="ZERO", nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery_7",
    **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nodes = nodes
    self.rows = rows
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.subfolder = f"{topology}_{index}_{mapping}"

    # Set ANALYSIS NAME ----
    self.analysis =  "{}_{}_{}".format(
      linkage.upper(), self.rows, self.nodes
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
        "../plots", self.common_path,
        self.analysis, mode, self.subfolder, discovery
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder, discovery
    )
    if "regions_path" in kwargs.keys():
      self.regions_path = kwargs["regions_path"]
    else:
      self.regions_path = join(
        "../CSV/Regions",
        "Table_areas_regions_09_2019.csv"
      )
    
    self.best_levels = np.zeros((max_iter, 2))
    self.R = np.zeros((max_iter, nodes, nodes))
    self.hierarchical_association = np.zeros((max_iter, nodes, nodes))
    self.labels = [""] * self.nodes
    self.colregion = pd.DataFrame()
  
  def create_plot_directory(self):
    Path(
      self.plot_path
    ).mkdir(exist_ok=True, parents=True)
    
  def create_pickle_directory(self):
    Path(
      self.pickle_path
    ).mkdir(exist_ok=True, parents=True)