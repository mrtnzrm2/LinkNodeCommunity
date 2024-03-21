import numpy as np
import os
from pathlib import Path
from various.network_tools import *

class SHUFFLE:
  def __init__(self, nodes, linkage : str,
      mode, iteration, nlog10=False, lookup=False, cut=False,
      topology="MIX", mapping="signed_trivial", index="Hellinger2", discovery="discovery_7", 
      version="PTN1200_recon2", structure="Cor"
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
    # Folder ----
    self.folder = "RAN"
    self.subject = "HCP"
    self.version = version
    self.structure = structure
    self.discovery = discovery
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set ANALYSIS NAME ----
    self.analysis = "{}_{}_{}".format(
      linkage.upper(), nodes, nodes
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
      self.structure,
      self.analysis, self.iter
    )
    self.plot_path = os.path.join(
      "../plots", self.common_path, mode,
      self.subfolder, discovery
    ) 
    self.csv_path = os.path.join(
      "../CSV", self.folder, self.random,
      self.subject, self.version,
      self.structure,
      self.iter
    )
    self.pickle_path = os.path.join(
      "../pickle", self.common_path, mode,
      self.subfolder, discovery
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