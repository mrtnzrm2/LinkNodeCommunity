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
    self.version = "40d91i"
    self.nature = "imputed"
    self.inj = "40i"
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "FLN"
    if "distance" in kwargs.keys():
      self.distance = kwargs["distance"]
    else: self.distance = "MAP3D"
    if "model" in kwargs.keys():
      self.model = kwargs["model"]
    else: self.model = ""
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature,
      self.distance, self.model,
      self.inj, "Consensus"
    )
    self.csv_path = "../CSV/MAC/40d91i/"
    self.distance_path = f"../CSV/MAC/40d91i/{self.distance}/"
    # Labels and regions ----
    self.labels_path = self.csv_path
  
  def create_csv_path(self):
      Path(
        self.csv_path
      ).mkdir(exist_ok=True, parents=True)

  def set_alpha(self, alpha):
    self.Alpha = np.sort(alpha)

  def set_beta(self, beta):
    self.Beta = np.sort(beta)

class MAC40i(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, imputation="RF", iteration=0, topology="MIX", mapping="R1", index="jacp", discovery="discovery_7", **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.imputation = imputation
    self.iteration = iteration
    self.discovery = discovery
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set attributes ----
    self.mode = mode
    # Get structure network ----
    self.A = self.get_structure()
    # Get network's spatial distances ----
    dist_dic = {
      "MAP3D" : self.get_distance_MAP3D,
      # "tracto16" : self.get_distance_tracto16
    }
    self.D = dist_dic[self.distance]()
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
    FLN = pd.read_csv(
      f"{self.csv_path}/FLN_40d91_{self.imputation}{self.iteration}.csv", #"Macaque_29x91_Arithmean_DBV23.45_GB_FIN_unnormalized.csv"
      # index_col=0
    )

    self.struct_labels = FLN.columns.to_numpy()
    self.rows, self.nodes = FLN.shape

    # np.savetxt(f"{self.csv_path}/labels40i.csv", self.struct_labels,  fmt='%s')

    return FLN.to_numpy()
    
  
  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "distances_macaque.csv")
    file = pd.read_csv(fname, index_col=0)

    labels = np.char.lower(file.columns.to_numpy(dtype=str))

    labels[labels == "35/36"] = "peri"
    labels[labels == "opal"] = "opai"
    labels[labels == "prost"] = "pro.st."
    labels[labels == "tea/m a"] = "tea/ma"
    labels[labels == "tea/m p"] = "tea/mp"
    labels[labels == "temp.pole"] = "pole"

    file.columns = labels
    file.index = labels

    # print([l for l in file.index if l not in self.struct_labels])
    # print([l for l in self.struct_labels if l not in file.index])

    D = file[self.struct_labels].loc[self.struct_labels]
    D = D.to_numpy()
    np.fill_diagonal(D, 0.)
    return D