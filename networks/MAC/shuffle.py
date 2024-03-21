import numpy as np
import os
from pathlib import Path
from various.network_tools import *

class SHUFFLE:
  def __init__(self, nodes, total_nodes, linkage : str,
      mode, iteration, nlog10=False, lookup=False, cut=False,
      topology="MIX", mapping="R1", index="jacp", discovery="discovery_7", 
      version=220830, structure="FLN", distance="tracto16"
    ) -> None:
    self.random = "shuffle"
    self.nodes = nodes
    self.iter = str(iteration)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.rows = total_nodes
    # Folder ----
    self.folder = "RAN"
    self.subject = "MAC"
    self.version = str(version)
    self.structure = structure
    self.distance = distance
    self.discovery = discovery
    self.model = ""
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set ANALYSIS NAME ----
    self.analysis = "{}_{}_{}".format(
      linkage.upper(), total_nodes, nodes
    )
    if nlog10:
      self.analysis = self.analysis + "_l10"
    if lookup:
      self.analysis = self.analysis + "_lup"
    if cut:
      self.analysis = self.analysis + "_cut"
    # Create common cut ----
    self.common_path = os.path.join(
      self.folder, self.random,
      self.subject, self.version,
      self.structure, self.distance,
      self.model,
      self.analysis, self.iter
    )
    self.plot_path = os.path.join(
      "../plots", self.common_path, mode,
      self.subfolder, discovery
    ) 
    self.csv_path = os.path.join(
      "../CSV", self.folder, self.random,
      self.subject, self.version,
      self.structure, self.distance,
      self.model,
      self.iter
    )
    self.dist_path = os.path.join(
      "../CSV", self.subject,
      self.version, self.structure,
      "original", self.distance,
      str(nodes)
    )
    self.pickle_path = os.path.join(
      "../pickle", self.common_path, mode,
      self.subfolder, discovery
    )
    # Labels and regions ----
    self.labels_path = self.dist_path
    self.regions_path = os.path.join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Overlap ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)

  def create_directory(self, path : str):
     Path(path).mkdir(exist_ok=True, parents=True)

  def create_plot_path(self):
    self.create_directory(self.plot_path)
  
  def create_pickle_path(self):
    self.create_directory(self.pickle_path)

  def create_csv_path(self):
    self.create_directory(self.csv_path) 
  
  def set_overlap(self, labels):
    self.overlap = labels
  
  def set_labels(self, labels):
    self.struct_labels = labels