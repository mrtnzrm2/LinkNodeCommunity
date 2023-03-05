from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *

class base:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    if "subject" in kwargs.keys():
      self.subject = kwargs["subject"]
    else: self.subject = "MAC"
    if "version" in kwargs.keys():
      self.version = str(kwargs["version"])
    else: self.version = str(220617)
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "FLN"
    if "nature" in kwargs.keys():
      self.nature = kwargs["nature"]
    else: self.nature = "original"
    if "distance" in kwargs.keys():
      self.distance = kwargs["distance"]
    else: self.distance = "tracto16"
    if "model" in kwargs.keys():
      self.model = kwargs["model"]
    else: self.model = ""
    if "inj" in kwargs.keys():
      self.inj = str(kwargs["inj"])
    else: self.inj = str(49)
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    if "b" in kwargs.keys():
      self.b = kwargs["b"]
    else: self.b = ""
    ### mu parameters ----
    self.Alpha = np.array([6])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,
      self.structure, self.nature,
      self.distance, self.model,
      self.inj
    )
    if "csv_path" in kwargs.keys():
      self.csv_path = join(
        kwargs["csv_path"], self.common_path
      )
    else:
      self.csv_path = join(
        "../CSV", self.common_path
      )
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

class MAC(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp",
    sln=False, **kwargs
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
    self.C, self.A = self.get_structure()
    # Get network's spatial distances ----
    self.D = self.get_distance()
    if sln: self.sln, self.supra, self.infra = self.get_sln()
    else: self.sln = np.nan
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
        "b_"+str(self.b)
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder,
      "b_"+str(self.b)
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
    if self.nature == "original":
      path = join(
        self.csv_path, "Count.csv"
      )
      C = pd.read_csv(path, index_col=0)
      clab = C.columns.to_numpy()
      rlab = np.array(C.index)
      nrlab = np.array([r for r in rlab if r not in clab])
      rlab = np.hstack([clab, nrlab])
      C = C.reindex(rlab).to_numpy()
      # Take out claustrum
      C = C[:-1, :]
      self.struct_labels = rlab[:-1]
      # self.struct_labels = rlab
      #
      self.struct_labels = np.array(self.struct_labels, dtype=str)
      self.struct_labels = np.char.lower(self.struct_labels)
      self.rows = C.shape[0]
      self.nodes = C.shape[1]
    elif self.nature == "imputation":
      path = join(
        self.csv_path, "fln.csv"
      )
      ##
      # C = pd.read_csv(path, index_col=0)
      C = pd.read_csv(path)
      ##
      self.struct_labels = C.columns.to_numpy().astype(str)
      self.struct_labels = np.char.lower(self.struct_labels)
      C = C.to_numpy()
      # Get network dimensions ----
      self.rows = C.shape[0]
      self.nodes = C.shape[1]
    else:
      raise ValueError("Data not understood.")
    return C,  column_normalize(C.copy())

  def get_distance(self):
    fname =  join(self.csv_path, "distance.csv")
    ##
    raw = pd.read_csv(fname, index_col=0)
    # raw = pd.read_csv(fname)
    ##
    raw.columns = np.char.lower(raw.columns.to_numpy().astype(str))
    raw = raw.set_index(np.array(raw.columns))
    # load areas definition
    raw = raw[self.struct_labels].reindex(self.struct_labels)
    D = raw.to_numpy()
    D[np.isnan(D)] = 0
    np.fill_diagonal(D, 0)
    return D

  def get_sln(self):
    ## Files ----
    fname = join(self.csv_path, "supraCount.csv")
    supra = pd.read_csv(fname, index_col=0)
    supra.columns = np.char.lower(supra.columns.to_numpy().astype(str))
    supra = supra.set_index(np.char.lower(supra.index.to_numpy().astype(str)))
    fname = join(self.csv_path, "infraCount.csv")
    infra = pd.read_csv(fname, index_col=0)
    infra.columns = np.char.lower(infra.columns.to_numpy().astype(str))
    infra = infra.set_index(np.char.lower(infra.index.to_numpy().astype(str)))
    ##
    supra = supra[self.struct_labels[:int(self.inj)]].reindex(self.struct_labels)
    infra = infra[self.struct_labels[:int(self.inj)]].reindex(self.struct_labels)
    supra = supra.to_numpy()
    infra = infra.to_numpy()
    ##
    sln = np.zeros(supra.shape)
    sln = supra / (supra + infra)
    sln[np.isnan(sln)] = 0
    return sln, supra, infra