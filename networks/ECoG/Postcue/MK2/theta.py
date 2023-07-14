from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *

class base:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    self.subject = "ECoG"
    self.version = "MK2"
    self.nature = "Postcue/theta"
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "GC"
    ### mu parameters ----
    self.Alpha = np.array([6, 25, 50])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.common_path = join(
      self.subject, self.structure, self.version,  self.nature
    )
    self.csv_path = "../CSV/MAC/ECoG/postcue/"
    self.distance_path = "../CSV/MAC/ECoG/"
    # Labels and regions ----
    self.labels_path = self.csv_path
  
  def create_csv_path(self):
      Path(
        self.csv_path
      ).mkdir(exist_ok=True, parents=True)

  def set_alpha(self, alpha):
    self.Alpha = alpha

  def set_beta(self, beta):
    self.Beta = beta

class MK2PostTheta(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set attributes ----
    self.mode = mode
    # Get structure network ----
    self.C = self.get_structure()
    # Get network's spatial distances ----
    self.D = self.get_distances()
    # Set ANALYSIS NAME ----
    # self.rows, self.nodes = 0, 0
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
        self.analysis, mode, self.subfolder
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder
    )
    if "regions_path" in kwargs.keys():
      self.regions_path = kwargs["regions_path"]
    else:
      self.regions_path = join(
        "../CSV/MAC/ECoG/",
        "sitepair_indices_mk1_mk2.txt"
      )
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
    file = pd.read_table(f"{self.csv_path}/mk2_gra_theta.txt", sep=" ", header=None)
    C = file.to_numpy()[:, :-1]
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    # Get labels ----
    labs = pd.read_table(f"{self.csv_path}/mk2_labels.txt", sep=" ", header= None)
    self.struct_labels = labs.to_numpy().ravel()
    return C
  
  def get_distances(self):
    file = pd.read_table(f"{self.distance_path}/mk2_postcue.txt", sep="\t", header=None)
    file = file.to_numpy()
    D = np.zeros((self.nodes, self.nodes))
    self.coords = {i : np.zeros(2) for i in np.arange(self.nodes)}
    for i in np.arange(self.nodes):
      self.coords[i][0] = file[i, 0]
      self.coords[i][1] = file[i, 1]
      for j in np.arange(i+1, self.nodes):
        D[i, j] = np.sqrt(np.power(file[i, 0] - file[j, 0], 2) + np.power(file[i, 1] - file[j, 1], 2))
        D[j, i] = D[i, j]
    return D