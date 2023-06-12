import numpy as np
import os
from networks.edrnet import EDR
from various.network_tools import *

class DISTBASE(EDR):
  def __init__(
    self, nodes, total_nodes, linkage,
    bin, mode, iter, nlog10=False, lookup=False, cut=False,
    topology="MIX", mapping="R1", index="jacp", lb=0.19, **kwargs
  ) -> None:
    super().__init__(nodes, lb=lb, **kwargs)
    self.random = "distbase"
    self.iter = str(iter)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.rows = 0
    self.subfolder = f"{topology}_{index}_{mapping}"
    # Define class attributes ----
    self.bin = bin
    # Set ANALYSIS NAME ----
    self.analysis = "{}_{}_{}".format(
      linkage.upper(), total_nodes, nodes
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
      self.structure, self.distance,
      self.model, "BIN_{}".format(bin),
      self.analysis, self.iter
    )
    self.plot_path = os.path.join(
      "../plots", self.common_path, mode,
      self.subfolder, "b_"+str(self.b)
    ) 
    self.csv_path = os.path.join(
      "../CSV", self.folder, self.random,
      self.subject, self.version,
      self.structure, self.distance,
      self.model, "BIN_{}".format(bin),
      self.iter
    )
    self.dist_path = os.path.join(
      "../CSV", self.subject,
      self.version, self.structure,
      "original", self.distance,
      str(nodes)
    )
    self.pickle_path = os.path.join(
      "../pickle", self.common_path, mode,
      self.subfolder, "b_"+str(self.b)
    )
    # Labels and regions ----
    self.labels_path = self.dist_path
    self.regions_path = os.path.join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Overlap ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)
    # dict sample_dist ----
    self.distbase_dict = {
      "EXPMLE" : self.random_net,
      "EXPTRUNC" : self.random_exp_trunc,
      "PARETO" : self.random_pareto,
      "PARETOTRUNC" : self.random_pareto_trunc,
      "LINEAR" : self.random_net,
      "LINEARTRUNC" : self.random_exp_trunc,
      "CONSTDEN" : self.random_const_net,
      "M" : self.random_net_M,
      "CONSTM": self.random_net_const_M,
      "CONSTPARETO" : self.random_const_pareto
    }

  def create_plot_path(self):
    self.create_directory(self.plot_path)
  
  def create_pickle_path(self):
    self.create_directory(self.pickle_path)

  def create_csv_path(self):
    self.create_directory(self.csv_path) 
  
  def set_overlap(self, labels):
    self.overlap = labels

  def get_structure(self):
    path = join(
      self.dist_path, "Count.csv"
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
    return C

  def binnarize(self, A):
    if A.shape[0] == A.shape[1]:
      dA = adj2df(A.copy())
    else:
      dA = A.copy()
    # Filter data ----
    dA = dA.loc[
      dA["source"] < dA["target"]
    ]
    # Get min and max distances
    wmin= np.nanmin(dA["weight"])
    wmax= np.nanmax(dA["weight"])
    # Create range boundaries ----
    b = np.linspace(wmin, wmax, self.bin)
    return b
  
  def random_pareto(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        # dD = dD.sort_values(by="weight", ascending=True)
        dD = dD.to_numpy()
        # Initiate samplenet ----
        from rand_network import sample_pareto
        NET = sample_pareto(
          dD, bin_ranges, self.bin,
          D.shape[0], args[0].shape[1], dD.shape[0], self.rho,
          self.lb, np.nanmin(D[D > 0])-1e-8, np.nanmax(D)
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(
              path, NET, delimiter=","
            )
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print(
          "NET density: {:.5f}".format(self.den(NET))
        )
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print(
        "NET density: {:.5f}".format(self.den(NET))
      )
    return NET
  
  def random_pareto_trunc(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        # dD = dD.sort_values(by="weight", ascending=True)
        dD = dD.to_numpy()
        # Initiate samplenet ----
        from rand_network import sample_pareto_trunc
        NET = sample_pareto_trunc(
          dD, bin_ranges, self.bin,
          D.shape[0], args[0].shape[1], dD.shape[0], self.rho,
          self.lb, np.min(D[D > 0]), np.nanmax(D)
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(
              path, NET, delimiter=","
            )
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print(
          "NET density: {:.5f}".format(self.den(NET))
        )
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print(
        "NET density: {:.5f}".format(self.den(NET))
      )
    return NET
  
  def random_const_pareto(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        # dD = dD.sort_values(by="weight", ascending=True)
        dD = dD.to_numpy()
        # Number of neurons ----
        counter = np.sum(args[0]).astype(int)
        # Initiate samplenet ----
        from rand_network import const_sample_pareto
        NET = const_sample_pareto(
          dD, bin_ranges, self.bin,
          D.shape[0], dD.shape[0], counter, self.rho,
          self.lb, np.nanmin(D[D > 0])-1e-8, np.nanmax(D),
          args[0].shape[0], args[0].shape[1]
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        print("NET count M: {}".format(self.count_M(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(path, NET, delimiter=",")
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET count M: {}".format(self.count_M(NET)))
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print("NET density: {:.5f}".format(self.den(NET)))
      print("NET Density: {:.5f}".format(self.Den(NET)))
      print("NET count M: {}".format(self.count_M(NET)))
    return NET

  def random_net(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        dD = dD.to_numpy()
        # Initiate samplenet ----
        from rand_network import sample_distbase
        NET = sample_distbase(
          dD, bin_ranges, self.bin,
          D.shape[0], args[0].shape[1], dD.shape[0],  self.rho,
          self.lb
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(path, NET, delimiter=",")
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print("NET density: {:.5f}".format(self.den(NET)))
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print("NET density: {:.5f}".format(self.den(NET)))
    return NET
  
  def random_exp_trunc(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        dD = dD.to_numpy()
        # Initiate samplenet ----
        from rand_network import sample_distbase_trunc
        NET = sample_distbase_trunc(
          dD, bin_ranges, self.bin,
          D.shape[0], args[0].shape[1], dD.shape[0],  self.rho,
          self.lb, np.min(D[D>0]), np.max(D)
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(path, NET, delimiter=",")
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print("NET density: {:.5f}".format(self.den(NET)))
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print("NET density: {:.5f}".format(self.den(NET)))
    return NET

  def random_net_M(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(dD)
        dD = dD.to_numpy()
        # Leaves!!!
        if np.sum(np.isnan(args[0])) > 0:
          counting_edges = np.sum(~np.isnan(args[0])).astype(int)
        else:
          counting_edges = np.sum(args[0] != 0).astype(int)
        # Initiate samplenet ----
        from rand_network import sample_distbase_M
        NET = sample_distbase_M(
          dD, bin_ranges, self.bin,
          args[0].shape[0], args[0].shape[1], dD.shape[0], counting_edges,
          self.lb
        )
        NET = np.array(NET)
        print("NET EC density: {:.5f}".format(self.den(NET)))
        print("NET M: {}".format(self.links_M(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(path, NET, delimiter=",")
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET M: {}".format(self.links_M(NET)))
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print("NET density: {:.5f}".format(self.den(NET)))
      print("NET M: {}".format(self.links_M(NET)))
    return NET

  def random_net_const_M(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(D)
        dD = dD.to_numpy()
        # Leaves!!!
        if np.sum(np.isnan(args[0])) > 0:
          leaves = np.sum(~np.isnan(args[0])).astype(int)
        else:
          leaves = np.sum(args[0] != 0).astype(int)
        # Number of neurons ----
        counter = np.sum(args[0]).astype(int)
        # Initiate samplenet ----
        from rand_network import const_sample_distbase_M
        NET = const_sample_distbase_M(
          dD, bin_ranges, self.bin,
          args[0].shape[0], args[0].shape[1], dD.shape[0], leaves,
          counter, self.lb
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        print("NET M: {}".format(self.links_M(NET)))
        print("NET count M: {}".format(self.count_M(NET)))
        if "on_save_csv" in kwargs.keys():
          if kwargs["on_save_csv"]:
            np.savetxt(path, NET, delimiter=",")
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print("NET density: {:.5f}".format(self.den(NET)))
        print("NET Density: {:.5f}".format(self.Den(NET)))
        print("NET M: {}".format(self.links_M(NET)))
        print("NET count M: {}".format(self.count_M(NET)))
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print("NET density: {:.5f}".format(self.den(NET)))
      print("NET Density: {:.5f}".format(self.Den(NET)))
      print("NET M: {}".format(self.links_M(NET)))
    return NET

  def random_const_net(self, D, *args, run=True, **kwargs):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if run:
      if not os.path.exists(path):
        # Prepare data ----
        dD = adj2df(D.copy())
        dD = dD.loc[
          dD["source"] < dD["target"]
        ]
        # Get bin ranges ----
        bin_ranges = self.binnarize(dD)
        dD = dD.to_numpy()
        # Initiate samplenet ----
        from rand_network import const_sample_distbase
        NET = const_sample_distbase(
          dD, bin_ranges, self.bin,
          D.shape[0], args[0].shape[1], dD.shape[0], self.counter,
          self.rho, self.lb
        )
        NET = np.array(NET)
        print("NET density: {:.5f}".format(
            self.den(NET)
          )
        )
        print("NET count: {}".format(
            self.count(NET)
          )
        )
        # np.savetxt(
        #   path, NET, delimiter=","
        # )
      else:
        NET = np.genfromtxt(path, delimiter=",")
        print(
          "NET density: {:.5f}".format(self.den(NET))
        )
        print("NET count: {}".format(
            self.count(NET)
          )
        )
    else:
      NET = np.genfromtxt(path, delimiter=",")
      print(
        "NET density: {:.5f}".format(self.den(NET))
      )
      print("NET count: {}".format(
          self.count(NET)
        )
      )
    return NET
  
  def set_labels(self, labels):
    self.struct_labels = labels