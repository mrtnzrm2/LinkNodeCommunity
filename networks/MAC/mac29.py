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
    self.subject = "MAC"
    self.version = "29d91"
    self.nature = "original"
    self.inj = "29"
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
    self.Alpha = np.array([6])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.common_path = join(
      self.subject, self.version,  self.structure, self.nature,
      self.distance, self.model,
      self.inj
    )
    self.csv_path = "../CSV/MAC/29d91/"
    self.distance_path = f"../CSV/MAC/29d91/{self.distance}/"
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

class MAC29(base):
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
    self.C, self.A = self.get_structure()
    self.SN, self.IN, self.SLN = self.get_sln_structure()
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

  def get_sln_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
    ## Areas to index
    tlabel = np.unique(file.TARGET).astype(str)
    inj = tlabel.shape[0]
    slabel = np.unique(file.SOURCE)
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1, dtype="<U21").astype(str)
    file["SOURCE_IND"] = match(file.SOURCE, slabel)
    file["TARGET_IND"] = match(file.TARGET, slabel)
    ## Average Count
    monkeys = np.unique(file.MONKEY)
    S = []
    I = []
    tid = np.unique(file.TARGET_IND)
    tmk = {t : [] for t in tid}
    for i, m in enumerate(monkeys):
      Sm = np.zeros((total_areas, inj))
      Im = np.zeros((total_areas, inj))
      data_m = file.loc[file.MONKEY == m]
      Sm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.SUPRAGRANULAR_NEURONS
      Im[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.INFRAGRANULAR_NEURONS
      S.append(Sm)
      I.append(Im)
      for t in np.unique(data_m.TARGET_IND): tmk[t].append(i)

    S = np.array(S)
    S[np.isnan(S)] = 0
    I = np.array(I)
    I[np.isnan(I)] = 0    

    SS = np.sum(S, axis=0)
    II = np.sum(I, axis=0)

    s = np.zeros((total_areas, inj))
    i = np.zeros((total_areas, inj))
    for t, mnk in tmk.items():
      s[:, t] = np.mean(S[mnk, :, t], axis=0)
      i[:, t] = np.mean(I[mnk, :, t], axis=0)
    
    A = np.zeros(SS.shape)
    A[SS > 0] = SS[SS > 0] / (SS[SS > 0] + II[SS > 0])

    slabel = np.char.lower(slabel)
    tlabel = np.char.lower(tlabel)

    original = np.array(["ins", "ento", "pole", "pir", "pi", "sub", "peri"])
    reference = np.array(["insula", "entorhinal", "temporal_pole", "piriform", "parainsula", "subiculum", "perirhinal"])

    slabel[match(original, slabel)] = reference

    # print([s for s in slabel if s not in self.struct_labels])
    s = pd.DataFrame(s, index=slabel, columns=tlabel)
    s = s[self.struct_labels[:self.nodes]].reindex(self.struct_labels).to_numpy()

    i = pd.DataFrame(i, index=slabel, columns=tlabel)
    i = i[self.struct_labels[:self.nodes]].reindex(self.struct_labels).to_numpy()

    A = pd.DataFrame(A, index=slabel, columns=tlabel)
    A = A[self.struct_labels[:self.nodes]].reindex(self.struct_labels)

    # A.to_csv(f"{self.csv_path}/sln_matrix.csv")
  # 
    # pd.DataFrame(s, index=slabel, columns=tlabel).to_csv(
    #   f"{self.csv_path}/smean_supra_neurons.csv"
    # )
    # pd.DataFrame(i, index=slabel, columns=tlabel).to_csv(
    #   f"{self.csv_path}/mean_infra_neurons.csv"
    # )

    return s.astype(float), i.astype(float), A.to_numpy().astype(float)
    

  def get_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/Neurons91x29_Arithmean_DBV23.45.csv", index_col=0, header=0)
    # print(file)
    col_labs = list(file.columns)
    row_labs = list(file.index)
    new_labs = col_labs + [r for r in row_labs if r not in col_labs]
    file = file.loc[new_labs, :]
    C = file.to_numpy()
    A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = new_labs
    self.struct_labels = np.char.lower(self.struct_labels)
    # np.savetxt(f"{self.csv_path}/labels29.csv", self.struct_labels,  fmt='%s')
    return C.astype(float), A.astype(float)

  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "DistanceMatrix Map3Dmars2019_91x91.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)
    # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # print(labs)
    # labs = [lab for lab in self.struct_labels if lab not in clabel]
    # print(labs)
    ## Rename areas from D to C
    D2C = {
      "pi" : "parainsula"
    }
    for key, val in D2C.items():
      clabel[clabel == key] = val
    # # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # # # print(labs)
    D = file.to_numpy()
    order = match(self.struct_labels, clabel)
    D = D[order, :][:, order]
    D = np.array(D)
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D