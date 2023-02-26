from os.path import join, exists
from os import getcwd
from pathlib import Path
from modules.hierarmerge import Hierarchy
from various.network_tools import *

class SCALEFREE:
  def __init__(self, iter, linkage, mode, nlog10=False, lookup=False, 
    cut=False, topology="", mapping="", index="", **kwargs
  ) -> None:
    self.nlog10 = nlog10
    self.lookup = lookup
    self.topology = topology
    self.mapping = mapping
    self.index = index
    self.cut = cut
    self.subfolder = f"{topology}_{index}_{mapping}"
    self.folder = "RAN"
    self.version = "scalefree"
    self.parameters = {
      "-N" : "100",
      "-k" : "20",
      "-maxk" : "30",
      "-mut" : "0.2",
      "-muw" : "0.2",
      "-beta" : "2",
      "-t1" : "2",
      "-t2" : "3"
    }
    if "parameters" in kwargs.keys():
      self.parameters = kwargs["parameters"]
    self.nodes = int(self.parameters["-N"])
    self.struct_labels = np.arange(self.nodes)
    # Set common_path from parameters ----
    # Set ANALYSIS NAME ----
    self.analysis = linkage.upper()
    if nlog10:
      self.analysis = self.analysis + "_l10"
    if lookup:
      self.analysis = self.analysis + "_lup"
    if cut:
      self.analysis = self.analysis + "_cut"
    # Create common path ----
    self.set_common_path()
    self.common_path = join(
      self.folder, self.version,
      self.common_path, str(iter)
    )
    # Save methods from net_tool ----
    self.column_normalize = column_normalize
    self.save_class = save_class
    self.read_class = read_class
    ## Define mu-score parameters ----
    self.Alpha = np.array([6])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Create paths ----
    self.wdn_path =  join(getcwd(), "cpp/WDN")
    self.plot_path = join(
      "../plots", self.common_path, 
      mode, self.subfolder
    )
    self.pickle_path = join(
      "../pickle", self.common_path, mode
    )
    self.regions_path = join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )

  def set_alpha(self, alpha):
    self.Alpha = alpha
  
  def set_beta(self, beta):
    self.Beta = beta

  def create_plot_path(self):
    Path(self.plot_path).mkdir(exist_ok=True, parents=True)

  def create_pickle_path(self):
    Path(self.pickle_path).mkdir(exist_ok=True, parents=True)
  
  def set_common_path(self):
    self.common_path = ""
    for key in self.parameters.keys():
      self.common_path = join(
        self.common_path,
        "{}_{}".format(key, self.parameters[key])
      )
    self.common_path = join(
      self.common_path, self.analysis
    )

  def paramters2list(self):
    fortunato = ""
    for k in self.parameters.keys():
      fortunato = "{} {} {}".format(
        fortunato, k, self.parameters[k]
      )
    return fortunato

  def set_data_measurements_zero(self, HH : Hierarchy, iter : int):
    H = get_H_from_BH_with_maxmu(HH)[
      ["K", "mu", "X", "D", "m", "ntrees"]
    ]
    H["data"] = ["0"] * H.shape[0]
    H["iter"] = [str(iter)] * H.shape[0]
    self.data_measures = pd.concat(
      [self.data_measures, H],
      ignore_index=True
    )

  def col_normalized_adj(self, on=True):
    if on:
      self.A = column_normalize(self.A)
      self.dA = adj2df(self.A.copy())
  
  def max_normalized_adj(self, on=True):
    if on:
      self.A = self.A / np.max(self.A)
      self.dA = adj2df(self.A.copy())

  def random_WDN(self, run=True, **kwargs):
    if run:
      if not exists(self.pickle_path + "/WDN.pk"):
        parameters = self.paramters2list()
        from subprocess import call
        call(
          join(self.wdn_path, "benchmark") + parameters,
          shell=True
        )
        self.dA = np.loadtxt(getcwd() + "network.dat")
        self.dA[:, :2] -= 1
        self.dA = pd.DataFrame(
          self.dA, columns=["source", "target", "weight"]
        )
        self.A = df2adj(self.dA.copy())
        self.labels = np.loadtxt(
          getcwd() + "community.dat"
        )[:, 1].astype(int) - 1
        if "on_save_pickle" in kwargs.keys():
          if kwargs["on_save_pickle"]:
            self.save_class(
              {
                "dA" : self.dA,
                "A" : self.A,
                "labels" : self.labels
              },
              self.pickle_path, "WDN"
            )
      else:
        WDN = self.read_class(self.pickle_path, "WDN")
        self.dA = WDN["dA"]
        self.A = WDN["A"]
        self.labels = WDN["labels"]
    else:
      WDN = self.read_class(self.pickle_path, "WDN")
      self.dA = WDN["dA"]
      self.A = WDN["A"]
      self.labels = WDN["labels"]
  
  def numeric_parameters(self):
    numeric_param = dict()
    parameters = self.parameters.copy()
    numeric_param["-N"] = int(parameters["-N"])
    for k in parameters.keys():
      if k == "-N": continue
      if k == "-on": continue
      if k == "-om": continue
      if k == "-nmin": continue
      if k == "-nmax": continue
      numeric_param[k] = float(parameters[k])
    if "-on" in parameters.keys():
      numeric_param["-on"] = int(parameters["-on"])
    else: numeric_param["-on"] = -1
    if "-om" in self.parameters.keys():
      numeric_param["-om"] = int(parameters["-om"])
    else: numeric_param["-om"] = -1
    if "-nmin" in self.parameters.keys():
      numeric_param["-nmin"] = int(parameters["-nmin"])
    else: numeric_param["-nmin"] = -1
    if "-nmax" in self.parameters.keys():
      numeric_param["-nmax"] = int(parameters["-nmax"])
    else: numeric_param["-nmax"] = -1
    return numeric_param

  def random_WDN_cpp(self, run=True, **kwargs):
    parameters = self.numeric_parameters()
    if run:
      if not exists(self.pickle_path + "/WDN_cpp.pk"):
        from WDN import WDN as wdn
        A = wdn(
          N = parameters["-N"],
          k = parameters["-k"],
          maxk = parameters["-maxk"],
          t1 = parameters["-t1"],
          t2 = parameters["-t2"],
          beta = parameters["-beta"],
          mut = parameters["-mut"],
          muw = parameters["-muw"],
          nmin = parameters["-nmin"],
          nmax = parameters["-nmax"]
        )
        self.dA = np.array(A.get_network())
        self.dA[:, :2] -= 1
        self.dA = pd.DataFrame(
          self.dA, columns=["source", "target", "weight"]
        )
        self.A = df2adj(self.dA.copy())
        self.labels = np.array(A.get_communities(), dtype=int)[:, 1] -1
        if "on_save_pickle" in kwargs.keys():
          if kwargs["on_save_pickle"]:
            self.save_class(
              {
                "dA" : self.dA,
                "A" : self.A,
                "labels" : self.labels
              },
              self.pickle_path, "WDN_cpp"
            )
      else:
        WDN = self.read_class(self.pickle_path, "WDN_cpp")
        self.dA = WDN["dA"]
        self.A = WDN["A"]
        self.labels = WDN["labels"]
    else:
      WDN = self.read_class(self.pickle_path, "WDN_cpp")
      self.dA = WDN["dA"]
      self.A = WDN["A"]
      self.labels = WDN["labels"]