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
    self.version = "40d91"
    self.nature = "original"
    self.inj = "40"
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
    self.csv_path = "../CSV/MAC/40d91/"
    self.distance_path = f"../CSV/MAC/40d91/{self.distance}/"
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

class MAC40(base):
  def __init__(
    self, linkage : str, mode : str, nlog10=True, lookup=False, 
    cut = False, topology="MIX", mapping="R1", index="jacp", discovery="discovery7", **kwargs
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
    self.C, self.CC, self.A = self.get_structure()
    self.SN, self.IN, self.SLN = self.get_sln_structure()
    self.get_beta_bbmodel()
    # Get network's spatial distances ----
    dist_dic = {
      "MAP3D" : self.get_distance_MAP3D,
      "tracto16" : self.get_distance_tracto16,
      "tracto91" : self.get_distance_tracto91
    }
    # self.D = self.get_distance_MAP3D_v2()
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
    file = pd.read_csv(f"{self.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
    ## Areas to index
    tlabel = np.unique(file.TARGET)
    inj = tlabel.shape[0]
    slabel = np.unique(file.SOURCE)
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1)
    file["SOURCE_IND"] = match(file.SOURCE, slabel)
    file["TARGET_IND"] = match(file.TARGET, slabel)
    ## Average Count
    monkeys = np.unique(file.MONKEY)
    C = []
    tid = np.unique(file.TARGET_IND)
    tmk = {t : [] for t in tid}
    for i, m in enumerate(monkeys):
      Cm = np.zeros((total_areas, inj))
      data_m = file.loc[file.MONKEY == m]
      Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
      C.append(Cm)
      for t in np.unique(data_m.TARGET_IND):
        tmk[t].append(i)
    C = np.array(C)
    C[np.isnan(C)] = 0
    CC = np.sum(C, axis=0).astype(int)

    c = np.zeros((total_areas, inj))
    for t, mnk in tmk.items(): c[:, t] = np.mean(C[mnk, :, t], axis=0)
    A = CC / np.sum(CC, axis=0)
    self.rows = c.shape[0]
    self.nodes = c.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)


    # wolfA = A.copy()
    # wolfA[wolfA != 0] = -1/np.log10(wolfA[wolfA != 0])
    # pd.DataFrame(wolfA, index=self.struct_labels, columns=self.struct_labels[:self.nodes]).to_csv(
    #   f"{self.csv_path}/boringness.csv"
    # )
    # np.savetxt(f"{self.csv_path}/labels.csv", self.struct_labels,  fmt='%s')

    return c.astype(float), CC.astype(float), A.astype(float)
  
  def get_distance_MAP3D_v2(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")

    ## Areas to index
    tlabel = np.unique(file.TARGET)
    inj = tlabel.shape[0]
    slabel = np.unique(file.SOURCE)
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1)
    file["SOURCE_IND"] = match(file.SOURCE, slabel)
    file["TARGET_IND"] = match(file.TARGET, slabel)

    ## Average Count
    monkeys = np.unique(file.MONKEY)
    D = []
    for i, m in enumerate(monkeys):
      Dm = np.zeros((total_areas, inj)) * np.nan
      data_m = file.loc[file.MONKEY == m]
      Dm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m["DISTANCE_(mm)"]
      D.append(Dm)

    D = np.array(D)
    D = np.nanmean(D, axis=0)
    D[np.isnan(D)] = 0
    return D


  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "DistanceMatrix Map3Dmars2019_91x91.csv")
    file = pd.read_csv(fname, index_col=0)
    
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    rlabel =file.index.to_numpy()
    rlabel = np.array([str(lab) for lab in rlabel])
    rlabel = np.char.lower(rlabel)

    # print(file)

    ## Rename areas from D to C
    D2C = {
      "perirhinal" : "peri",
      "entorhinal" : "ento",
      "subiculum" : "sub",
      "temporal_pole" : "pole",
      "insula" : "ins",
      "piriform" : "pir"
    }
    for key, val in D2C.items():
      clabel[clabel == key] = val
      rlabel[rlabel == key] = val
    # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # print(labs)
    D = file.to_numpy()
    D = pd.DataFrame(D, index=rlabel, columns=clabel)[self.struct_labels].reindex(self.struct_labels).to_numpy()

    np.fill_diagonal(D, 0.)
    D = D.astype(float)

    return D
  
  def get_distance_tracto16(self):
    fname =  join(self.distance_path, "106x106_DistanceMatrix.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # print(labs)
    # ## Rename areas from D to C
    D2C = {
      "subi" : "sub",
      "insula" : "ins",
      "v1c" : "v1",
      "v2c" : "v2",
      "v3c" : "v3",
      "v4c" : "v4",
      "mtc" : "mt"
    }
    for key, val in D2C.items():
      clabel[clabel == key] = val
    D = file.to_numpy()
    order = match(self.struct_labels, clabel)
    D = D[order, :][:, order]
    D = np.array(D)
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D
  
  def get_distance_tracto91(self):
    fname =  join(self.distance_path, "Macaque_TractoDist_91x91_220418.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    clabel[clabel == "insula"] = "ins" 
    clabel[clabel == "subi"] = "sub" 

    file.columns = clabel
    file.index = clabel
  
    D91 = file[self.struct_labels].loc[self.struct_labels].to_numpy()
    D3d = self.get_distance_MAP3D()

    def predict_missing_values(x, y):
      from scipy.stats import linregress
      res = linregress(x, y)
      return res.intercept, res.slope

    d91 = np.triu(D91, 1).ravel()
    d3d = np.triu(D3d, 1).ravel()


    d3d = d3d[(~np.isnan(d91)) & (d91 < 80)]
    d91 = d91[(~np.isnan(d91)) & (d91 < 80)]

    # print(np.sum(np.isnan(D91))-91)

    b, m = predict_missing_values(d3d, d91)

    # miss = np.isnan(D91)
    # print(np.nanmax(D91))
    D91[np.isnan(D91)] = m * D3d[np.isnan(D91)] + b
    D91[D91 > 80] = m * D3d[D91 > 80] + b

    # print(D91[miss])
    np.fill_diagonal(D91, 0.)

    # print(np.sum(np.isnan(D91)))

    # raise ValueError("")
    return D91.astype(float)
  
  def get_sln_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
    ## Areas to index
    tlabel = np.unique(file.TARGET).astype(str)
    inj = tlabel.shape[0]
    slabel = np.unique(file.SOURCE)
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1).astype(str)
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

    s = pd.DataFrame(s, index=slabel, columns=tlabel)
    s = s[self.struct_labels[:self.nodes]].reindex(self.struct_labels).to_numpy()

    i = pd.DataFrame(i, index=slabel, columns=tlabel)
    i = i[self.struct_labels[:self.nodes]].reindex(self.struct_labels).to_numpy()

    A = pd.DataFrame(A, index=slabel, columns=tlabel).reindex(self.struct_labels)

    # A.to_csv(f"{self.csv_path}/sln_matrix.csv")

    # pd.DataFrame(s, index=slabel, columns=tlabel).to_csv(
    #   f"{self.csv_path}/mean_supra_neurons.csv"
    # )
    # pd.DataFrame(i, index=slabel, columns=tlabel).to_csv(
    #   f"{self.csv_path}/mean_infra_neurons.csv"
    # )
    return s.astype(float), i.astype(float), A.to_numpy().astype(float)
  
  def get_beta_bbmodel(self):
    beta = pd.read_csv(f"{self.csv_path}/sln/sln_beta_coefficients_40.csv", index_col=1).reindex(self.struct_labels[:self.nodes])
    self.beta = beta["beta"].to_numpy()