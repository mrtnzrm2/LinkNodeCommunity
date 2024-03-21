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
    self.version = "53d106"
    self.nature = "original"
    self.inj = "53"
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
    if "b" in kwargs.keys():
      self.b = kwargs["b"]
    else: self.b = ""
    ### mu parameters ----
    self.Alpha = np.array([6, 20])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature,
      self.distance, self.model,
      self.inj
    )
    self.csv_path = "../CSV/MAC/53d91/"
    self.distance_path = f"../CSV/MAC/53d91/{self.distance}/"
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

class SUPRA53(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery_3", **kwargs
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
        self.analysis, mode, self.subfolder,
        f"b_{self.b}", discovery
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder,
      f"b_{self.b}", discovery
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
    file = pd.read_csv(f"{self.csv_path}/Macaque91x53SupraGranularMat_230719.csv", index_col=0)
    ## Areas to index
    tlabel = file.columns.to_numpy()
    slabel = file.index.to_numpy()
    slabel1 = [lab for lab in slabel if lab not in tlabel and lab != "Claustrum"]
    slabel = np.array(list(tlabel) + slabel1)
    ## Average Count
    C = file.loc[slabel, tlabel]
    # print(C)
    C = C.to_numpy(dtype=float)
    # A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)

    self.labels = self.struct_labels
    # np.savetxt(f"{self.csv_path}/labels53.csv", self.struct_labels,  fmt='%s')

    return C.astype(float)

  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "Macaque91x91DistanceMat_230719.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    file.columns = clabel
    file.index = [s.lower() for s in file.index]

    file = file.loc[self.labels, self.labels]
    # print(self.labels)
    # print(file)
    # print(file)
    D = file.to_numpy()
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D
  
class INFRA53(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery_3", **kwargs
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
        self.analysis, mode, self.subfolder,
        f"b_{self.b}", discovery
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder,
      f"b_{self.b}", discovery
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
    file = pd.read_csv(f"{self.csv_path}/Macaque91x53InfraGranularMat_230719.csv", index_col=0)
    ## Areas to index
    tlabel = file.columns.to_numpy()
    slabel = file.index.to_numpy()
    slabel1 = [lab for lab in slabel if lab not in tlabel and lab != "Claustrum"]
    slabel = np.array(list(tlabel) + slabel1)
    ## Average Count
    C = file.loc[slabel, tlabel]
    # print(C)
    C = C.to_numpy(dtype=float)
    # A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)

    self.labels = self.struct_labels
    # np.savetxt(f"{self.csv_path}/labels53.csv", self.struct_labels,  fmt='%s')

    return C.astype(float)

  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "Macaque91x91DistanceMat_230719.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    file.columns = clabel
    file.index = [s.lower() for s in file.index]

    file = file.loc[self.labels, self.labels]
    # print(file)
    D = file.to_numpy()
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D