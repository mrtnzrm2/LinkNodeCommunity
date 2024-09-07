from os.path import join
import os
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *

def normalize_MUS_name(name):
    """convert the given area name (row or column header in raw csv)
    to a standardized identifier"""
    name = name.lower()    
    return name

class Nansafe:
    """helper class to create a with scope where operations on nans and infs do not generate runtime errors/warnings"""

    def __enter__(self):
        self.old_settings = np.seterr(divide='ignore', invalid='ignore')

    def __exit__(self, type, value, traceback):
        np.seterr(**self.old_settings) # restore

class base:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    self.subject = "MUS"
    self.version = "19d47"
    self.nature = "original"
    self.inj = "19"
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "FLNe"
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
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature,
      self.distance, self.model,
      self.inj
    )
    self.csv_path = "../CSV/MUS/19d47/"
    self.distance_path = f"../CSV/MUS/19d47/{self.distance}/"
    # Labels and regions ----
    self.labels_path = self.csv_path
  
  def create_csv_path(self):
      Path(
        self.csv_path
      ).mkdir(exist_ok=True, parents=True)

class MUS19(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="trivial", index="Helligner2", discovery="discovery_7", **kwargs
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
    self.areas, self.lookup_label = self.makeOrder()
    self.A, self.C = self.get_structure()
    self.CC = self.C.copy()
    # Get network's spatial distances ----
    dist_dic = {
      "MAP3D" : self.get_distance_MAP3D,
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
        "b_"+str(self.b), discovery
      )
    self.pickle_path = join(
      "../pickle", self.common_path,
      self.analysis, mode, self.subfolder,
      "b_"+str(self.b), discovery
    )
    if "regions_path" in kwargs.keys():
      self.regions_path = kwargs["regions_path"]
    else:
      self.regions_path = join(
        "../CSV/Regions",
        "Mouse_Cortical_Regions_complete.csv"
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

  def makeOrder(self, filename="Mouse-Database_download_2024.csv"):
    """Compute the common ordering of the cortical areas."""
    
    # load the newer dataset
    fname = os.path.join(self.csv_path, filename)
    raw = pd.read_csv(fname)
    
    areas = { str(item).lower() for item in raw['Source area'].values }
    
    known = { str(item).lower() for item in raw['Target area'].values }
    
    unknown = areas - known
    
    # compose and sort within groups
    
    tgt1 = list(sorted(known))
    tgt2 = list(sorted(unknown))
    
    names = tgt1 + tgt2
        
    lookup = { names[i] : i for i in range(len(names)) }
    
    return names, lookup

  def get_structure(self):
    # Get structure ----
    fname = os.path.join(self.csv_path, "Mouse-Database_download_2024.csv")
    raw = pd.read_csv(fname)

    # FLN matrix
    n = len(self.areas)
    m = len(set(raw["Target area"].values)) # should be 19
    CC = np.zeros((n, m))
    FLN = np.zeros((n, m))
    C = np.zeros((n, m)) # count of data records
    
    mouses = set(raw["Mouse"].values)
    for mouse in mouses:
        data = raw[raw["Mouse"]==mouse]
        
        # one mouse could have received multiple target injections
        targets = { str(val) for val in data["Target area"].values }
        
        for target in targets:
            v = self.lookup_label[target.lower()]
            proc = data[data["Target area"]==target]
            
            # compute the total neuron count at this target
            total = np.nansum(proc["Neurons"].values)
            if total <= 0:
                break
            
            # compute FLN values and store them
            for i in range(len(proc)):
                src = str(proc["Source area"].iloc[i])
                u = self.lookup_label[src.lower()]
                
                fln = float(proc["Neurons"].iloc[i]) / total
                CC[u,v] += float(proc["Neurons"].iloc[i])
                FLN[u,v] += fln
                C[u,v] += 1
    
    with Nansafe():
        FLN /= C

    self.nodes = m
    self.rows = n
    self.struct_labels = self.areas
    zero = np.logical_or(np.isnan(FLN), np.isinf(FLN))
    FLN[zero] = 0.
    CC[zero] = 0.

    self.struct_labels = np.array([s.lower() for s in self.struct_labels])
    # np.savetxt(f"{self.csv_path}/labels19.csv", self.struct_labels,  fmt='%s')
    return FLN, CC

  def get_distance_MAP3D(self):

    fname = os.path.join(self.csv_path, "distances.csv")
    raw = pd.read_csv(fname)

    # distance matrix
    n = len(self.areas)
    
    D = np.zeros((n, n)) # distance values
    # loop over the input raw data
    for i in range(n):
        tgtName = raw.columns[i+1]
        tgt = normalize_MUS_name(tgtName)
        col = self.lookup_label[tgt]
        ser = raw[tgtName]

        for j in range(n):
            src = raw['Unnamed: 0'].iloc[j]
            src = normalize_MUS_name(src)
            row = self.lookup_label[src]
                    
            if row==col: 
                D[row,col] = 0
                continue
            
            D[row,col] = float(ser.iloc[j])

    D = np.array(D)
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D
  
  def get_ODR_structure(self):
    file = pd.read_csv(os.path.join(self.csv_path, "SourceData_Fig3.csv"))
    pathway_var = file["pathway"]
    pathway_var = [(s.split(" to ")[0], s.split(" to ")[1]) for s in pathway_var]
    file["SOURCE_IND"] = [s for s, t in pathway_var]
    file["TARGET_IND"] = [t for s, t in pathway_var] 
    file["SOURCE_IND"] = np.char.lower([s[1:] for s in file["SOURCE_IND"]])
    file["TARGET_IND"] = np.char.lower([s[:-1] for s in file["TARGET_IND"]])

    labels_order_paper_DSouzza = np.char.lower(
      ["V1", "LM", "RL", "AL", "A", "PM", "P", "LI", "AM", "POR"]
    )

    file["SOURCE_IND"] = pd.Categorical(file["SOURCE_IND"], labels_order_paper_DSouzza)
    file["TARGET_IND"] = pd.Categorical(file["TARGET_IND"], labels_order_paper_DSouzza)

    # return file.pivot("SOURCE_IND", "TARGET_IND", "mean"), labels_order_paper_DSouzza