from os.path import join
import os
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *

class base:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    self.subject = "HCP"
    self.version = "PTN1200_recon2"
    self.nature = "original"
    self.nodetimeseries = "25"
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "Cor"
    if "nodetimeseries" in kwargs.keys():
      self.nodetimeseries = kwargs["nodetimeseries"]
    else: self.structure = "25"
    if "model" in kwargs.keys():
      self.model = kwargs["model"]
    else: self.model = ""
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = "MIX"
    ### mu parameters ----
    self.Alpha = np.array([6, 20])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature, self.model,
      self.nodetimeseries
    )
    self.csv_path = "../CSV/HCP/"
    # Labels and regions ---
    self.labels_path = self.csv_path
  
  def create_csv_path(self):
      Path(
        self.csv_path
      ).mkdir(exist_ok=True, parents=True)

  def set_alpha(self, alpha):
    self.Alpha = np.sort(alpha)

  def set_beta(self, beta):
    self.Beta = np.sort(beta)

class HCP(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery_3",
     architecture="product-positive", undirected=0, subject_id=None, **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.discovery = discovery
    self.subject_id = subject_id
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set attributes ----
    self.mode = mode
    # Get structure network ----
    self.A = self.get_structure()
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
        self.analysis, mode, self.subfolder, discovery, f"undirected_{undirected}",
        architecture
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder, discovery, f"undirected_{undirected}",
      architecture
    )
    if subject_id:
      self.plot_path = join(self.plot_path, subject_id)
      self.pickle_path = join(self.pickle_path, subject_id)
    # Path pictures ----
    self.picture_path = f"../CSV/HCP/images_{self.nodetimeseries}"
    # entropy ----
    self.entropy = np.array([0, 0, 0])
    # NOCS labels ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)
    # NOCS labels and memberships ----
    self.data_nocs = {}
  
  def create_plot_directory(self):
    Path(
      self.plot_path
    ).mkdir(exist_ok=True, parents=True)
    
  def create_pickle_directory(self):
    Path(
      self.pickle_path
    ).mkdir(exist_ok=True, parents=True)

  def get_structure(self):
    # Get structure ----
    # file = pd.read_table(f"../CSV/HCP/netmats2_{self.nodetimeseries}.txt", sep=" ").to_numpy()
    # file = np.mean(file, axis=0).reshape(int(self.nodetimeseries), int(self.nodetimeseries))
    # np.fill_diagonal(file, 0)

    # np.savetxt(f"../CSV/HCP/netmats2_{self.nodetimeseries}_mean.txt", file)

    if not self.subject_id:
      file = pd.read_table(f"../CSV/HCP/netmats2_{self.nodetimeseries}_mean.txt", sep=" ", header=None).to_numpy()
    else:
      line = os.listdir(f"../CSV/HCP/nodetimeseries_{self.nodetimeseries}")
      line = np.array([l.split(".")[0] for l in line])
      line = np.where(line == self.subject_id)[0]
      file = pd.read_table(f"../CSV/HCP/netmats2_{self.nodetimeseries}.txt", sep=" ").to_numpy()
      file = file[line].reshape(int(self.nodetimeseries), int(self.nodetimeseries))
      np.fill_diagonal(file, 0)

      
    self.rows, self.nodes = file.shape
    self.struct_labels = np.arange(self.nodes).astype(str)

    # perm = np.random.permutation(self.nodes)
    # file = file[perm, :][:, perm]
    # self.struct_labels = self.struct_labels[perm]

    self.labels = self.struct_labels
    return file.astype(float)