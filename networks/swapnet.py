import os
from os.path import join
import pandas as pd
import numpy as np
import numpy.typing as npt
from networks.edrnet import EDR
from various.network_tools import column_normalize

class SWAPNET(EDR):
  def __init__(
    self, nodes : int, total_number_of_nodes : int, linkage : str, mode : str, iteration : int,
    nlog10=False, lookup=False, cut=False,
    mapping="R1", topology="MIX", index="jacp", discovery="discovery_5", **kwargs
  ) -> None:
    super().__init__(nodes, **kwargs)
    self.random = "swaps"
    self.iter = iteration
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.discovery = discovery
    self.subfolder = f"{topology}_{index}_{mapping}"
    ## the distance matrix ----
    self.structure_path = os.path.join(
      "../CSV", self.subject,
      self.version, self.structure,
      self.nature, self.distance,
    )
    if self.nature == "original":
      self.structure_path = os.path.join(
        self.structure_path, str(self.nodes)
      )
    elif self.nature == "imputation":
      self.structure_path = os.path.join(
        self.structure_path, self.model, str(self.nodes)
      )
    # Get structure network ----
    self.C, self.A = 0, 0
    self.rows = total_number_of_nodes
    # Set ANALYSIS NAME ----
    self.analysis = linkage.upper() + "_{}_{}".format(
      self.rows, self.nodes
    )
    if nlog10:
      self.analysis = self.analysis + "_l10"
    if lookup:
      self.analysis = self.analysis + "_lup"
    if cut:
      self.analysis = self.analysis + "_cut"
    # Directory details ----
    self.common_path = os.path.join(
      self.folder, self.random,
      self.subject, self.version,
      self.structure, self.distance,
      self.model, self.analysis, str(self.iter),
    )
    self.plot_path = os.path.join(
      "../plots", self.common_path, mode,
      self.subfolder, "b_"+str(self.b), discovery
    )
    self.csv_path = os.path.join(
      "../CSV", self.folder, self.random,
      self.subject, self.version,
      self.structure, self.distance,
      self.model, str(self.iter) 
    )
    self.pickle_path = os.path.join(
      "../pickle", self.common_path, mode, self.subfolder,
      "b_"+str(self.b), discovery
    )
    # Labels and regions ----
    self.labels_path = self.structure_path
    self.regions_path = os.path.join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Get network's spatial distances ----
    self.D = 0
    # Base clustering attributes ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)

  def create_plot_path(self):
    self.create_directory(self.plot_path)
  
  def create_pickle_path(self):
    self.create_directory(self.pickle_path)

  def create_csv_path(self):
    self.create_directory(self.csv_path)

  def set_colregion(self, colregion):
    self.colregions = colregion
  
  def get_distance(self):
    fname = join(
      self.structure_path, "distance.csv"
    )
    ##
    raw = pd.read_csv(fname, index_col=0)
    # raw = pd.read_csv(fname)
    ##
    raw.columns = np.char.lower(raw.columns.to_numpy().astype(str))
    raw = raw.set_index(raw.columns)
    # load areas definition
    raw = raw[self.struct_labels].reindex(self.struct_labels)
    D = raw.to_numpy()
    D[np.isnan(D)] = 0
    np.fill_diagonal(D, 0)
    return D

  def random_one_k(self, run=True, swaps=10000, **kwargs):
    if self.nature == "original":
      path = os.path.join(
        self.csv_path, "Count.csv"
      )
      if run:
        if not os.path.exists(path):
          from rand_network import swap_one_k
          t = swap_one_k(
            self.A, self.rows, self.nodes, swaps
          )
          self.A = np.array(t)
          print(
            "Density: {}".format(
              self.den(self.A[:self.nodes, :self.nodes])
            )
          )
          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path, self.A, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())
      else:
        self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())
    elif self.nature == "imputation":
      path = os.path.join(
        self.csv_path, "fln.csv"
      )
      if run:
        if not os.path.exists(path):
          from rand_network import swap_one_k
          t = swap_one_k(
            self.A, self.rows, self.nodes, 100000
          )
          self.A = np.array(t)
          print(
            "Density: {}".format(
              self.den(self.A[:self.nodes, :self.nodes])
            )
          )
          # print(
          #   "Count: {}".format(self.count(self.C))
          # )
          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path, self.A, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path, delimiter=",")
      else:
        self.A = np.genfromtxt(path, delimiter=",")

  def random_dir_weights(self, run=True, swaps=10000, **kwargs):
    if self.nature == "original":
      path = os.path.join(
        self.csv_path, "Count.csv"
      )
      if run:
        if not os.path.exists(path):
          from rand_network import swap_dir_weights
          t = swap_dir_weights(
            self.A, self.rows, self.nodes, swaps
          )
          self.A = np.array(t)

          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path, self.A, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())
      else:
        self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())

  def A_random_one_k(self, A : npt.NDArray, swaps=100000):
      n, m = A.shape
      from rand_network import swap_one_k
      t = swap_one_k(
        self.A, n, m, swaps
      )
      return np.array(t)
         

  def random_one_k_TWOMX(self, SM : npt.NDArray, run=True, swaps=100000, **kwargs):
    if self.nature == "original":
      path_A = os.path.join(
        self.csv_path, "A.csv"
      )
      path_B = os.path.join(
        self.csv_path, "B.csv"
      )
      if run:
        if not os.path.exists(path_A):
          from rand_network import swap_one_k_TWOMX

          # if not (self.A.shape == SM.shape):
          #   raise RuntimeError(">>> The two matrices do not have the same dimensions.")
          # t = np.array(
          #   swap_one_k_TWOMX(
          #     self.A[:self.nodes, :][:, :self.nodes],
          #     SM[:self.nodes, :][:, :self.nodes],
          #     self.nodes, self.nodes, swaps
          #   )
          # )

          t = np.array(
            swap_one_k_TWOMX(
              self.A, SM, self.rows, self.nodes, swaps
            )
          )

          self.A = t[0]
          self.B = t[1]

          if np.sum(np.isnan(self.B)) > 0:
            raise RuntimeError("Swapping instroced nans")
          
          print(
            "Density edge-complete: {}".format(
              self.den(self.A[:self.nodes, :self.nodes])
            )
          )
          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path_A, self.A, delimiter=","
              )
              np.savetxt(
                path_B, self.B, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path_A, delimiter=",")
          self.B = np.genfromtxt(path_B, delimiter=",")
      else:
        self.A = np.genfromtxt(path_A, delimiter=",")
        self.B = np.genfromtxt(path_B, delimiter=",")


  def random_one_k_dense(self, run=True, swaps=100000, **kwargs):
    if self.nature == "original":
      path = os.path.join(
        self.csv_path, "Count.csv"
      )
      if run:
        if not os.path.exists(path):
          print(">>>>", self.rows, self.nodes)
          from rand_network import swap_one_k_dense
          t = swap_one_k_dense(
            self.A, self.rows, self.nodes, swaps
          )
          self.A = np.array(t)
          print(
            "Density: {}".format(
              self.den(self.A[:self.nodes, :self.nodes])
            )
          )
          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path, self.A, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())
      else:
        self.A = np.genfromtxt(path, delimiter=",")
        # self.A = column_normalize(self.C.copy())
    elif self.nature == "imputation":
      path = os.path.join(
        self.csv_path, "fln.csv"
      )
      if run:
        if not os.path.exists(path):
          from rand_network import swap_one_k_dense
          t = swap_one_k_dense(
            self.A, self.rows, self.nodes, 100000
          )
          self.A = np.array(t)
          print(
            "Density: {}".format(
              self.den(self.A[:self.nodes, :self.nodes])
            )
          )
          # print(
          #   "Count: {}".format(self.count(self.C))
          # )
          if "on_save_csv" in kwargs.keys():
            if kwargs["on_save_csv"]:
              np.savetxt(
                path, self.A, delimiter=","
              )
        else:
          self.A = np.genfromtxt(path, delimiter=",")
      else:
        self.A = np.genfromtxt(path, delimiter=",")

