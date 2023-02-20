# Standard libs ----
from pathlib import Path
import numpy as np
import pandas as pd
from os.path import join
# Personal libs ----
from various.network_tools import *
from various.similarity_indices import NT
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
    self.subfolder = H.subfolder
    # Get labels from L ----
    self.labels = L.labels[:self.nodes]
    # Create data ----
    self.data = pd.DataFrame()
    # Node community membership ----
    self.labels_nc = pd.DataFrame()
    # Overlap ----
    self.data_overlap = pd.DataFrame()
    # Homonegenity ----
    self.data_homoegeity = pd.DataFrame()
    self.set_data_homogeneity_one()
    # Measurements ----
    self.data_measures = pd.DataFrame()
    self.set_data_measurements_one(H)
    # K ,R ----
    self.kr = pd.DataFrame()
    self.set_kr_one(H)
    # Set save_class as method ----
    self.minus_one_Dc = H.minus_one_Dc
    self.save_class = save_class
    self.read_class = read_class
    # Set stats ----
    self.stats = H.stats
    self.stats["data"] = "1"

  def set_subfolder(self, subfolder):
    self.subfolder = subfolder

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
          ["K", "mu", "X", "D", "m", "ntrees"]
        ]
      ],
      ignore_index=True
    )
    self.data_measures["data"] = ["1"] * self.data_measures.shape[0]
  
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

  def set_data_homogeneity_one(self):
    NT_vector = np.zeros(self.nodes)
    for i in np.arange(self.nodes):
      NT_vector[i] = NT(self.R, i, axis=1)
    self.data_homoegeity = pd.concat(
      [
        self.data_homoegeity,
        pd.DataFrame(
          {
            "area" : self.labels,
            "TNH" : NT_vector,
            "data" : ["1"] * self.nodes
          }
        )
      ],
      ignore_index=True
    )
  
  def set_data_homogeneity_zero(self, R):
    NT_vector = np.zeros(self.nodes)
    for i in np.arange(self.nodes):
      NT_vector[i] = NT(R, i, axis=1)
    self.data_homoegeity = pd.concat(
      [
        self.data_homoegeity,
        pd.DataFrame(
          {
            "area" : self.labels,
            "TNH" : NT_vector,
            "data" : ["0"] * self.nodes
          }
        )
      ],
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

  def set_nodes_labels_single(self, H : Hierarchy, score):
    # Set labels from data ----
    k, r = get_best_kr(score, H)
    labels =  get_labels_from_Z(H.Z, r)
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
    self.minus_one_Dc(d)
    return d["id"].to_numpy()

  def set_iter(self, it):
    self.iter = it

  def set_nmi_nc(self, l2, score):
    labels = self.labels_nc.id.loc[
      self.labels_nc.score == score
    ].to_numpy()
    #create subdata ----
    subdata = pd.DataFrame(
      {
        "NMI" : [
          AD_NMI_label(labels, l2)
        ],
        "c" : ["node community"],
        "iter" : [self.iter],
        "score" : [score]
      }
    )
    # Merge with data ----
    self.data = pd.concat(
      [self.data, subdata],
      ignore_index=True
    )

  def set_pickle_path(self, H : Hierarchy):
     # Get pickle location ----
    pickle_path = H.pickle_path.split("/")
    self.pickle_path = ""
    for i in np.arange(len(pickle_path)):
      self.pickle_path = join(
        self.pickle_path, pickle_path[i]
      )
      if pickle_path[i] == self.subfolder: break
    self.pickle_path = join(
      self.pickle_path, H.mode, H.subfolder
    )
    Path(self.pickle_path).mkdir(exist_ok=True, parents=True)

  def set_plot_path(self, H : Hierarchy):
    plot_path = H.plot_path.split("/")
    self.plot_path = ""
    for i in np.arange(len(plot_path)):
      self.plot_path = join(
        self.plot_path, plot_path[i]
      )
      if plot_path[i] == self.subfolder: break
    self.plot_path = join(
      self.plot_path, H.mode, H.subfolder
    )
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