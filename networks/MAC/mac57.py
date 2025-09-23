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
    self.version = "57d106"
    self.nature = "original"
    self.inj = "57"
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

    if "root_path" in kwargs.keys():
      self.csv_path = kwargs["root_path"] + "/CSV/MAC/57d106_230605/"
      self.distance_path = f"{self.csv_path}/{self.distance}/"

    else:
      self.csv_path = "../CSV/MAC/57d106_230605/"
      self.distance_path = f"../CSV/MAC/57d106_230605/{self.distance}/"

    # self.csv_path = "../CSV/MAC/57d106_230605/"
    # self.distance_path = f"../CSV/MAC/57d106_230605/{self.distance}/"
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

class MAC57(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery_7", **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.discovery = discovery
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Set attributes ----
    self.mode = mode
    # Get structure network ----
    self.C = self.get_structure()
    self.CC, self.A = self.get_summer_counts()
    # Get network's spatial distances ----
    dist_dic = {
      # "MAP3D" : self.get_distance_MAP3D,
      "tracto16" : self.get_distance_tracto16
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
    self.root_path  = ""
    if "root_path" in kwargs.keys():
      self.root_path = kwargs["root_path"]

      self.plot_path = join(
        self.root_path , "plots", self.common_path,
        self.analysis, mode, self.subfolder,
        "b_"+str(self.b)
      ) 

      self.pickle_path = join(
        self.root_path , "pickle", self.common_path,
        self.analysis, mode, self.subfolder,
        "b_"+str(self.b)
      )

      self.regions_path = join(
        self.root_path , "CSV/Regions",
        "Table_areas_regions_09_2019.csv"
      )
      
    else:
      self.plot_path = join(
        "../plots", self.common_path,
        self.analysis, mode, self.subfolder,
        "b_"+str(self.b)
      )

      self.pickle_path = join(
        "../pickle", self.common_path,
        self.analysis, mode, self.subfolder,
        "b_"+str(self.b)
      )
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
    file = pd.read_csv(f"{self.csv_path}/CountMatrix_Averaged_57areas_230605.csv", index_col=0)
    ## Areas to index
    tlabel = file.columns.to_numpy()
    slabel = file.index.to_numpy()
    slabel1 = [lab for lab in slabel if lab not in tlabel and lab != "Claustrum"]
    slabel = np.array(list(tlabel) + slabel1)
    ## Average Count
    C = file[tlabel].loc[slabel]
    C = C.to_numpy(dtype=float)
    # A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)
    # Permute network
    # import random
    # from datetime import datetime
    # random.seed(datetime.now().timestamp())
    # self.perm = np.random.permutation(np.arange(self.nodes))
    # self.perm2 = list(self.perm) + list(np.arange(self.nodes, self.rows))
    # C = C[self.perm2, :][:, self.perm]
    # self.struct_labels = self.struct_labels[self.perm2]
    ##
    self.labels = self.struct_labels
    # np.savetxt(f"{self.csv_path}/labels57.csv", self.struct_labels,  fmt='%s')
    # NET.struct_labels = NET.struct_labels[perm2]
    return C.astype(float)

  def get_summer_counts(self):
    file = pd.read_csv(f"{self.csv_path}/CountMatrix_Summed_57areas_220830.csv", index_col=0)
    file.columns = np.char.lower(file.columns.to_numpy(dtype=str))
    file.index = np.char.lower(file.index.to_numpy(dtype=str))
    file = file[self.struct_labels[:self.nodes]].loc[self.struct_labels]
    CC = file.to_numpy(dtype=float)
    A = CC / np.sum(CC, axis=0)
    return CC, A
  
  def get_distance_tracto16(self):
    fname =  join(self.distance_path, "106x106_DistanceMatrix.csv")
    file = pd.read_csv(fname, index_col=0)
    # print(file.columns.to_numpy())
    file.columns = np.char.lower(file.columns.to_numpy(dtype=str))
    file.index = np.char.lower(file.index.to_numpy(dtype=str))
    D = file[self.struct_labels].loc[self.struct_labels]
    D = D.to_numpy()
    np.fill_diagonal(D, 0.)
    return D

  # def get_distance_MAP3D(self):
  #   fname =  join(self.distance_path, "DistanceMatrix Map3Dmars2019_91x91.csv")
  #   file = pd.read_csv(fname, index_col=0)
  #   clabel =file.columns.to_numpy()
  #   clabel = np.array([str(lab) for lab in clabel])
  #   clabel = np.char.lower(clabel)
  #   ## Rename areas from D to C
  #   D2C = {
  #     "perirhinal" : "peri",
  #     "entorhinal" : "ento",
  #     "subiculum" : "sub",
  #     "temporal_pole" : "pole",
  #     "insula" : "ins",
  #     "piriform" : "pir"
  #   }
  #   for key, val in D2C.items():
  #     clabel[clabel == key] = val
  #   # labs = [lab for lab in clabel if lab not in self.struct_labels]
  #   # print(labs)
  #   D = file.to_numpy()
  #   order = match(self.struct_labels, clabel)
  #   D = D[order, :][:, order]
  #   D = np.array(D)
  #   np.fill_diagonal(D, 0.)
  #   D = D.astype(float)
  #   return D