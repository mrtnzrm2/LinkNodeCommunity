import numpy as np
from networks.scalefree import SCALEFREE
from modules.hierarmerge import Hierarchy
from various.network_tools import *

class OVERLAPPING(SCALEFREE):
  def __init__(
    self, iter, linkage, mode,
    nlog10=False, lookup=False, cut=False,
    topology="MIX", mapping="R1", index="jacp", **kwargs
  ) -> None:
    super().__init__(
      iter, linkage, mode, nlog10=nlog10, lookup=lookup,
      topology=topology, index=index, mapping=mapping,
      cut=cut, **kwargs
    )

  def read_data_multicolumns(self):
    # Transform community.dat into list ----
    with open("community.dat") as file_in:
      lines = []
      for line in file_in:
        line = line.split(" ")
        line = line[:-1]
        line2 = []
        for l in line:
          if "\t" in l:
            l = l.split("\t")
          for i in l:
            line2.append(int(i))
        lines.append(line2)
    # Create labels and overlap ----
    labels = np.zeros(self.nodes)
    overlap = dict()
    for i, line in enumerate(lines):
      labels[i] = line[1] -1
      if len(line) > 2:
        overlap[line[0] - 1] = [c - 1 for c in line[1:]]
    return labels, overlap

  def read_dat_multicolumns_cpp(self, lab):
    # Create labels and overlap ----
    labels = np.zeros(self.nodes)
    overlap = dict()
    for i, line in enumerate(lab):
      labels[i] = line[1] - 1
      if len(line) > 2:
        overlap[line[0] - 1] = [c - 1 for c in line[1:]]
    return labels, overlap

  def random_WDN_overlap(self, run=True, **kwargs):
    if run:
      if not exists(self.pickle_path + "/WDN.pk"):
        parameters = self.paramters2list()
        from subprocess import call
        call(
          join(self.wdn_path, "benchmark") + parameters,
          shell=True
        )
        self.dA = np.loadtxt("network.dat")
        self.dA[:, :2] -= 1
        self.dA = pd.DataFrame(
          self.dA, columns=["source", "target", "weight"]
        )
        self.A = df2adj(self.dA.copy())
        self.labels, self.overlap = self.read_data_multicolumns()
        print(self.overlap)
        if "on_save_pickle" in kwargs.keys():
          if kwargs["on_save_pickle"]:
            self.save_class(
              {
                "dA" : self.dA,
                "A" : self.A,
                "labels" : self.labels,
                "overlap" : self.overlap
              },
              self.pickle_path, "WDN"
            )
      else:
        WDN = self.read_class(self.pickle_path, "WDN")
        self.dA = WDN["dA"]
        self.A = WDN["A"]
        self.labels = WDN["labels"]
        self.overlap = WDN["overlap"]
        print(self.overlap)
    else:
      WDN = self.read_class(self.pickle_path, "WDN")
      self.dA = WDN["dA"]
      self.A = WDN["A"]
      self.labels = WDN["labels"]
      self.overlap = WDN["overlap"]
      print(self.overlap)

  def random_WDN_overlap_cpp(self, run=True, **kwargs):
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
          on = parameters["-on"],
          om = parameters["-om"],
          nmin = parameters["-nmin"],
          nmax = parameters["-nmax"]
        )
        self.dA = np.array(A.get_network())
        if len(self.dA) == 0:
          self.dA = np.array([np.nan])
          self.A = np.array([np.nan])
          self.labels = np.array([np.nan])
          self.overlap = dict()
        else:
          self.dA[:, :2] -= 1
          self.dA = pd.DataFrame(
            self.dA, columns=["source", "target", "weight"]
          )
          self.A = df2adj(self.dA.copy())
          self.labels, self.overlap = self.read_dat_multicolumns_cpp(
            A.get_communities()
          )
          print(self.overlap)
          if "on_save_pickle" in kwargs.keys():
            print("\n\t**** Network saved in pickle format****\n")
            if kwargs["on_save_pickle"]:
              self.save_class(
                {
                  "dA" : self.dA,
                  "A" : self.A,
                  "labels" : self.labels,
                  "overlap" : self.overlap
                },
                self.pickle_path, "WDN_cpp"
              )
      else:
        WDN = self.read_class(self.pickle_path, "WDN_cpp")
        self.dA = WDN["dA"]
        self.A = WDN["A"]
        self.labels = WDN["labels"]
        self.overlap = WDN["overlap"]
        print(self.overlap)
    else:
      WDN = self.read_class(self.pickle_path, "WDN_cpp")
      self.dA = WDN["dA"]
      self.A = WDN["A"]
      self.labels = WDN["labels"]
      self.overlap = WDN["overlap"]
      print(self.overlap)

  def set_colregion(self, colregion):
    self.colregion = colregion

  def stats_max_slc(self, data):
    subdata = data.groupby(["nodes"])["size"].max()
    mean = subdata.mean()
    std = subdata.std()
    return mean, std

  def stats_max_minus_one(self, data, labels):
    catch_nodes = []
    subdata = data.groupby(["nodes"]).max()
    max_size = subdata["size"]
    nodes = subdata.index
    for i, nd in enumerate(nodes):
      sub =  data.loc[
        (data["nodes"] == nd) &
        (data["ids"] == -1)
      ]
      if sub.shape[0] == 0: continue
      x = sub["size"].iloc[0]
      if x == max_size.iloc[i]:
        y = np.where(labels == nd)[0][0]
        catch_nodes.append(y)
    return catch_nodes
  
  def discover_overlap_nodes(self, H, K):
    overlap = np.zeros(self.nodes)
    labels = self.colregion.labels[:self.nodes]
    for k in K:
      from scipy.cluster.hierarchy import cut_tree
      ## Without -1 ----
      dA = H.dA.copy()
      # Cut tree ----
      dA["id"] = cut_tree(
        H.H, n_clusters=k
      ).reshape(-1)
      H.minus_one_Dc(dA)
      aesthetic_ids(dA)
      dA = dA.loc[dA["id"] != -1]
      # Get lc sizes for each node ----
      data = bar_data(
        dA, self.nodes, labels, norm=True
      )
      # Get stats ----
      mean_s, std_s = self.stats_max_slc(data)
      # Grouped data ----
      size =  data.groupby("nodes")["size"].max()
      x = size < mean_s - 2 * std_s
      x = [
        np.where(labels == nd)[0][0] for i, nd in enumerate(x.index) if x.iloc[i]
      ]
      overlap[x] += 1
      ## Max -1 ----
      dA = H.dA.copy()
      dA["id"] = cut_tree(
        H.H, n_clusters=k
      ).reshape(-1)
      H.minus_one_Dc(dA)
      aesthetic_ids(dA)
      # Get lc sizes for each node ----
      data = bar_data(
        dA, self.nodes, labels, norm=True
      )
      # Geta stats ----
      minus_one_nodes = self.stats_max_minus_one(data, labels)
      if len(minus_one_nodes) > 0:
        overlap[minus_one_nodes] += 1
    return np.where(overlap > 0)[0]

  def overlap_score(self, H : Hierarchy, K, labels, on=False):
    if on:
      if 1 in K and len(K) == 1: return np.nan, np.nan
      ALL = set(H.colregion.labels[:H.nodes])
      PRED = set(H.get_ocn(labels))
      gt = self.overlap.keys()
      GT = set([str(g) for g in gt])
      # sensitivity = tp / P
      TP = GT.intersection(PRED)
      tp = len(TP)
      FN = GT - PRED
      fn = len(FN)
      sen = tp / (tp + fn)
      # specificity = TN / N
      TN = ALL - GT
      tn = len(TN)
      FP = PRED - GT
      fp = len(FP)
      sep = tn / (tn + fp)
      print(
        "sen = {:.7f}\nsp = {:.7f}".format(sen, sep)
      )
      return sen, sep
    else:
      return -1, -1

  def omega_index(self, node_partition, noc_covers, node_labels, on=False):
    if on:
      from various.omega import Omega
      gt_noc_cover = self.overlap
      gt_node_partition = self.labels
      gt_covers = omega_index_format(gt_node_partition, gt_noc_cover, node_labels)
      pred_covers = omega_index_format(node_partition, noc_covers, node_labels)
      omega = Omega(pred_covers, gt_covers).omega_score
      print(f"Omega index: {omega:.4f}")
    else: omega = -1
    return omega

  def overlap_score_discovery(self, K : int, nocs, labels, on=False):
    if on:
      if K == 1: return np.nan, np.nan
      # Ground-truth partition prep. ----
      gt = self.overlap.keys()
      GT = set([str(g) for g in gt])
      ## Sensitivity/Specificity ----
      ALL = set(labels)
      PRED = set(nocs)
      # sensitivity = tp / P
      TP = GT.intersection(PRED)
      tp = len(TP)
      FN = GT - PRED
      fn = len(FN)
      sen = tp / (tp + fn)
      # specificity = TN / N
      TN = ALL - GT
      tn = len(TN)
      FP = PRED - GT
      fp = len(FP)
      sep = tn / (tn + fp)
      print(
        "sen = {:.4f}\nsp = {:.4f}".format(sen, sep)
      )
      return sen, sep
    else:
      return -1, -1
      

