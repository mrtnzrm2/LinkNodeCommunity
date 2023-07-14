# Standard libs ----
from pathlib import Path
import numpy as np
import pandas as pd
from os.path import join
from various.omega import Omega
# Personal libs ----
from various.network_tools import *
from modules.hierarmerge import Hierarchy
from modules.colregion import colregion

class HRH:
  def __init__(self, H: Hierarchy, L : colregion) -> None:
    # Load attributes from H ----
    self.linkage = H.linkage
    self.nodes = H.nodes
    self.Z = H.Z
    self.dA = H.dA
    self.R = H.R
    self.rcover = H.cover
    self.subfolder = H.subfolder
    # Get labels from L ----
    self.labels = L.labels[:self.nodes]
    # Create data ----
    self.data = pd.DataFrame()
    # Node community membership ----
    self.labels_nc = pd.DataFrame()
    # Cover ----
    self.cover = {}
    # Overlap ----
    self.data_overlap = pd.DataFrame()
    # Partition, labels, & covers ----
    self.set_various_labels(H)
    # Measurements ----
    self.data_measures = pd.DataFrame()
    self.set_data_measurements_one(H)
    # K ,R ----
    self.kr = pd.DataFrame()
    self.set_kr_one(H)
    # Entropy ----
    self.node_entropy = pd.DataFrame()
    self.link_entropy = pd.DataFrame()
    self.set_entropy_one(H.entropy)
    # Association matrice ----
    self.association_one = {}
    self.association_zero = {}
    for key in self.rcover.keys():
      self.set_association_one(key)
      self.association_zero[key] = np.zeros((self.nodes, self.nodes))
    # Set save_class as method ----
    self.save_class = save_class
    self.read_class = read_class
    # Set stats ----
    self.stats = H.stats
    self.stats["data"] = "1"

  def set_subfolder(self, subfolder):
    self.subfolder = subfolder

  def set_association_one(self, score):
    self.association_one[score] = np.zeros((self.nodes, self.nodes))
    for key, nodes in self.rcover[score].items():
      for i in np.arange(len(nodes)):
        for j in np.arange(i+1, len(nodes)):
          x = np.where(self.labels == nodes[i])[0][0]
          y = np.where(self.labels == nodes[j])[0][0]
          if key != -1:
            self.association_one[score][x, y] += 1
            self.association_one[score][y, x] += 1
          else:
            self.association_one[score][x, y] -= 1
            self.association_one[score][y, x] -= 1
  
  def set_association_zero(self, score, cover : dict):
    for key, nodes in cover.items():
      for i in np.arange(len(nodes)):
        for j in np.arange(i+1, len(nodes)):
          x = np.where(self.labels == nodes[i])[0][0]
          y = np.where(self.labels == nodes[j])[0][0]
          if key != -1:
            self.association_zero[score][x, y] += 1
            self.association_zero[score][y, x] += 1
          else:
            self.association_zero[score][x, y] -= 1
            self.association_zero[score][y, x] -= 1

  def set_various_labels(self, NET_H : Hierarchy):
    for i in np.arange(NET_H.kr.shape[0]):
      r = NET_H.kr.R.iloc[i]
      score = NET_H.kr.score.iloc[i]
      rlabels = get_labels_from_Z(self.Z, r)
      overlap_labels = NET_H.overlap.labels.loc[NET_H.overlap.score == score].to_numpy()
      self.set_overlap_data_one(overlap_labels, score)
      self.set_nodes_labels(rlabels, score)
      # Cover
      self.set_cover_one(NET_H.cover[score], score)

  def set_entropy_one(self, s):
    dim = s[0].shape[1]
    self.node_entropy = pd.concat(
      [
        self.node_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[0].ravel(), s[1].ravel()]), "data" : ["1"] * 4 * dim,
            "c" : ["node_hierarchy"] * 2 * dim  + ["node_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4), "iter" : [np.nan] * 4 * dim
          } 
        )
      ], ignore_index=True
    )
    dim = s[2].shape[1]
    self.link_entropy = pd.concat(
      [
        self.link_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[2].ravel(), s[3].ravel()]), "data" : ["1"] * 4 * dim,
            "c" : ["link_hierarchy"] * 2 * dim  + ["link_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4), "iter" : [np.nan] * 4 * dim
          }
        )
      ], ignore_index=True
    )

  def set_entropy_zero(self, s):
    dim = s[0].shape[1]
    self.node_entropy = pd.concat(
      [
        self.node_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[0].ravel(), s[1].ravel()]), "data" : ["0"] * 4 * dim,
            "c" : ["node_hierarchy"] * 2 * dim  + ["node_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4),
            "iter" : [(self.iter+1)] * dim + [-(self.iter+1)] * dim + [(self.iter+1)] * dim + [-(self.iter+1)] * dim
          } 
        )
      ], ignore_index=True
    )
    dim = s[2].shape[1]
    self.link_entropy = pd.concat(
      [
        self.link_entropy,
        pd.DataFrame(
          {
            "S" : np.hstack([s[2].ravel(), s[3].ravel()]), "data" : ["0"] * 4 * dim,
            "c" : ["link_hierarchy"] * 2 * dim  + ["link_hierarch_H"] * 2 * dim,
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "level" : np.tile(np.arange(dim, 0, -1), 4),
            "iter" : [(self.iter+1)] * dim + [-(self.iter+1)] * dim + [(self.iter+1)] * dim + [-(self.iter+1)] * dim
          }
        )
      ], ignore_index=True
    )

  def set_kr_one(self, H : Hierarchy):
    self.kr = pd.concat(
      [self.kr, H.kr],
      ignore_index=True
    )
    self.kr["data"] = ["1"] * H.kr.shape[0]

  def set_kr_zero(self, HH : Hierarchy):
    df = HH.kr.copy()
    df["data"] = ["0"] * df.shape[0]
    self.kr = pd.concat(
      [self.kr, df],
      ignore_index=True
    )

  def set_data_measurements_one(self, H : Hierarchy):
    self.data_measures = pd.concat(
      [
        self.data_measures,
        get_H_from_BH_with_maxmu(H)[
          ["K", "X", "D", "S", "SD"]
        ]
      ],
      ignore_index=True
    )
    self.data_measures["data"] = ["1"] * self.data_measures.shape[0]
  
  def set_data_measurements_zero(self, HH : Hierarchy, iter : int):
    H = get_H_from_BH_with_maxmu(HH)[
      ["K", "X", "D", "S", "SD"]
    ]
    H["data"] = ["0"] * H.shape[0]
    H["iter"] = [str(iter)] * H.shape[0]
    self.data_measures = pd.concat(
      [self.data_measures, H],
      ignore_index=True
    )
  
  def set_overlap_data_one(self, overlap, score):
    w = [i for i, nd in enumerate(self.labels) if nd in overlap]
    subdata = pd.DataFrame(
      {
        "Areas" : self.labels[w],
        "data" : ["1"] * len(w),
        "score" : [score] * len(w)
      }
    )
    self.data_overlap = pd.concat(
      [self.data_overlap, subdata],
      ignore_index=True
    )

  def set_cover_one(self, cover, score):
    self.cover[score] = cover

  def set_nodes_labels(self, labels, score):
    sublabels = pd.DataFrame(
      {
        "id" : labels,
        "score" : [score] * len(labels)
      }
    )
    self.labels_nc = pd.concat(
      [self.labels_nc, sublabels],
      ignore_index=True
    )

  def set_stats(self, H : Hierarchy):
    stats = H.stats
    stats["data"] = "0"
    self.stats = pd.concat(
      [
        self.stats,
        stats
      ],
      ignore_index=True
    )

  def set_labels_average(self, WSBM, K, R, on=False):
    if on:
      print("Set data labels for average-linkage")
      self.labels_nc = WSBM.pick_pair(K, R)

  def minus_one_labels(self, dA, labels):
    d = pd.DataFrame(
      {
        "source" : dA["source"],
        "target" : dA["target"],
        "id" : labels
      }
    )
    minus_one_Dc(d)
    return d["id"].to_numpy()

  def set_iter(self, it):
    self.iter = it

  def set_clustering_similarity(self, l2, cover, score):
    labels = self.labels_nc.id.loc[self.labels_nc.score == score].to_numpy()
    #create subdata ----
    subdata = pd.DataFrame(
      {
        "sim" : ["NMI", "OMEGA"],
        "values" : [
          AD_NMI_overlap(labels, l2, self.cover[score], cover),
          omega_index(cover, self.cover[score])
        ],
        "c" : ["node community"] * 2,
        "iter" : [self.iter] * 2,
        "score" : [score] * 2
      }
    )
    # Merge with data ----
    self.data = pd.concat(
      [self.data, subdata],
      ignore_index=True
    )

  def set_pickle_path(self, H : Hierarchy, bias=0):
     # Get pickle location ----
    pickle_path = H.pickle_path.split("/")
    self.pickle_path = ""
    for i in np.arange(len(pickle_path)):
      self.pickle_path = join(
        self.pickle_path, pickle_path[i]
      )
      if pickle_path[i] == H.analysis: break
    self.pickle_path = join(
      self.pickle_path, H.mode, H.subfolder
    )
    self.pickle_path = join(self.pickle_path, f"b_{bias}")
    Path(self.pickle_path).mkdir(exist_ok=True, parents=True)

  def set_plot_path(self, H : Hierarchy, bias=0):
    plot_path = H.plot_path.split("/")
    self.plot_path = ""
    for i in np.arange(len(plot_path)):
      self.plot_path = join(
        self.plot_path, plot_path[i]
      )
      if plot_path[i] == H.analysis: break
    self.plot_path = join(
      self.plot_path, H.mode, H.subfolder
    )
    self.plot_path = join(self.plot_path, f"b_{bias}")
    Path(self.plot_path).mkdir(exist_ok=True, parents=True)

  def set_overlap_data_zero(self, overlap, score):
    if len(overlap) > 0:
      w = [i for i, nd in enumerate(self.labels) if nd in overlap]
      subdata = pd.DataFrame(
        {
          "Areas" : self.labels[w],
          "data" : ["0"] * len(w),
          "score" : [score] * len(w)
        }
      )
      self.data_overlap = pd.concat(
        [self.data_overlap, subdata],
        ignore_index=True
      )
    else: print(f"No OCN with the {score} score")