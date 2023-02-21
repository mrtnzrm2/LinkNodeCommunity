# Standard libs ----
import numpy as np
import pandas as pd
import os
#  Personal libs ----
from various.network_tools import *
from modules.colregion import colregion
from modules.simanalysis import Sim
import process_hclust as ph
from la_arbre_a_merde import noeud_arbre

class Hierarchy(Sim):
  def __init__(
    self, G, A, R, D, nodes, linkage, mode, lookup=0
  ):
    # Initialize Sim ---
    super().__init__(
      nodes, A, R, D, mode,
      topology=G.topology, index=G.index,
      lookup=lookup
    )
    # Set parameters
    self.linkage = linkage
    self.cut = G.cut
    self.alpha = G.Alpha
    self.beta = G.Beta
    self.pickle_path = G.pickle_path
    self.plot_path = G.plot_path
    self.subfolder = G.analysis
    # Compute similarity matrix ----
    self.similarity_by_feature_cpp()
    # Compute distance matrix ----
    self.dist_mat = self.linksim_matrix
    self.dist_mat[self.dist_mat == 0] = np.nan
    self.dist_mat = 1 - self.dist_mat
    self.dist_mat[np.isnan(self.dist_mat)] = np.nanmax(self.dist_mat) + 100
    # Compute hierarchy ----
    self.H = self.get_hierarchy()
    # Network to edgelist of EC ----
    self.dA = adj2df(
      self.A.copy()[:self.nodes, :]
    )
    ## Take out zeros ----
    self.dA = self.dA.loc[self.dA.weight != 0]
    ## Get some functions
    self.minus_one_Dc = minus_one_Dc 
    self.Dc_id = Dc_id
    ## Stats ----
    self.stat_cor()
    # Overlaps ----
    self.overlap = pd.DataFrame()
    # KR ----
    self.kr = pd.DataFrame()

  def set_kr(self, k, r, score=""):
    self.kr = pd.concat(
      [
        self.kr,
        pd.DataFrame(
          {
            "K" : [k],
            "R" : [r],
            "score" : [score]
          }
        )
      ], ignore_index=True
    )

  def get_hierarchy(self):
    from scipy.cluster.hierarchy import linkage
    return linkage(
      condense_madtrix(
        self.dist_mat
      ),
      self.linkage
    )

  #** Feature methods:
  def mu(self, id, n, beta):
    # LC frequencies ----
    from collections import Counter
    id_counter =  dict(Counter(id))
    # Decreasingly sort frequencies ----
    id_counter = {
      k: v for k, v in sorted(
        id_counter.items(), key=lambda item: item[1],
        reverse=True
      )
    }
    ## Get keys ----
    id_keys = list(id_counter.keys())
    ## Initilaize mu_score ----
    mu_score = 0
    # Start loop ----
    size = len(id_keys)
    if size >= n:
      for i in np.arange(n - 1):
        for j in np.arange(i + 1, n):
          D = id_counter[id_keys[j]] / id_counter[id_keys[i]]
          if D > beta:
            mu_score += D * (
              id_counter[id_keys[j]] + id_counter[id_keys[i]]
            ) / self.leaves
          else:
            mu_score -= D * (
              id_counter[id_keys[j]] + id_counter[id_keys[i]]
            ) / self.leaves
      mu_score /= 0.5 * n * (n - 1)
    else:
      for i in np.arange(size - 1):
        for j in np.arange(i + 1, size):
          D = id_counter[id_keys[j]] / id_counter[id_keys[i]]
          if D > beta:
            mu_score += D * (
              id_counter[id_keys[j]] + id_counter[id_keys[i]]
            ) / self.leaves
          else:
            mu_score -= D * (
              id_counter[id_keys[j]] + id_counter[id_keys[i]]
            ) / self.leaves
      mu_score /= 0.5 * n * (n - 1)
    return mu_score

  def get_ncs(self, df):
    uni_ids = np.unique(df["id"].to_numpy())
    nc = np.zeros(2)
    nc[0] = np.copy(self.nodes)
    for id in uni_ids:
      src = df.loc[df["id"] == id, "source"].to_numpy()
      tgt = df.loc[df["id"] == id, "target"].to_numpy()
      nodes_c = np.unique(
        np.concatenate((src, tgt), axis=0)
      )
      m = len(src)
      if m > 1 and len(nodes_c) > 2:
        inter = np.intersect1d(src, tgt)
        if len(inter) > 1:
          nc[0] -= len(inter)
          nc[0] += 1
        nc[1] += 1
    return nc

  def tree_height(self):
    dis_H = self.H[:, 2]
    nH = len(dis_H)
    height = np.zeros(nH)
    height[0] = dis_H[0] / 2
    for i in np.arange(1, nH):
      height[i] = height[i - 1] + (dis_H[i] - dis_H[i - 1]) / 2
    return height

  def H_features_cpp(self, linkage, alpha, beta, cut=False):
    # Get network dataframe ----
    dA =  self.dA.copy()
    # Run process_hclust_fast.cpp ----
    features = ph.process_hclust_fast(
      self.leaves,
      self.dist_mat,
      dA["source"].to_numpy(),
      dA["target"].to_numpy(),
      self.nodes,
      linkage,
      cut,
      alpha,
      beta
    )
    features = np.array(features)
    return features

  def area(self, x, y):
    x_1 = x[1:]
    x_0 = x[:(len(x)-1)]
    h = (x_1 - x_0) / 2
    y_0 = y[:(len(y) - 1)]
    y_1 = y[1:]
    v = (y_1 - y_0)/2
    a = np.nansum(v * h)
    a = np.abs(a)
    return a

  def BH_features_cpp(self):
    print("Computing features over mu-score space")
    # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # Create BH ----
    self.BH = []
    # Loop over mu-score parameters ----
    for alpha in self.alpha:
      for beta in self.beta:
        print(
          "Alpha: {} and Beta: {:.4f}". format(
            alpha, beta
          )
        )
        BH = self.H_features_cpp(
          linkage, alpha, beta, cut=self.cut
        )[:, :10]
        #  mu = BH[:, 4] / self.area(BH[:, 0], BH[:, 4])
        self.BH.append(
          pd.DataFrame(
            {
              "alpha" : [alpha] * BH.shape[0],
              "beta" : [np.round(beta, 4)] * BH.shape[0],
              "K" : BH[:, 0],
              "height" : BH[:, 1],
              "NAC" : BH[:, 2],
              "NEC" : BH[:, 3],
              "mu" : BH[:, 4],
              "D" : BH[:, 5],
              "ntrees": BH[:, 6],
              "X" : BH[:, 7],
              "m" : BH[:, 8],
              "xm" : BH[:, 9]
            }
          )
        )

  def la_abre_a_merde_cpp(self, features):
    # Get network dataframe ----
    dA =  self.dA.copy()
     # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 1
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # Run la_abre_a_merde_vite ----
    if features.K.iloc[-1] != 1:
      features = pd.concat(
        [
          features,
          pd.DataFrame(
            {
              "n" : [features.n.iloc[0]],
              "beta" : [features.beta.iloc[0]],
              "K" : 1,
              "height" : [features.height.iloc[-1] * 1.01],
              "NAC" : [1],
              "NEC" : [1],
              "mu" : [np.nan],
              "D" : [features.D.iloc[-1] * 1.01],
              "ntrees" : [0]
            }
          )
        ],
        ignore_index=True
      )
    NH = noeud_arbre(
      self.dist_mat,
      dA["source"].to_numpy(),
      dA["target"].to_numpy(),
      features["K"].to_numpy().astype(int),
      features["height"].to_numpy(),
      features["NEC"].to_numpy().astype(int),
      self.nodes,
      self.leaves,
      linkage,
      features.shape[0]
    )
    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)
  
  def stat_cor(self):
    # Copy target and source similarity matrices ----
    dIN = self.source_sim_matrix
    dIN[np.isnan(dIN)] = np.nanmin(dIN) - 1
    dIN = adj2df(dIN)
    dIN = dIN.loc[
      dIN["source"] < dIN["target"], "weight"
    ].to_numpy().ravel()
    dOUT = self.target_sim_matrix
    dOUT[np.isnan(dOUT)] = np.nanmin(dOUT) - 1
    dOUT = adj2df(dOUT)
    dOUT = dOUT.loc[
      dOUT["source"] < dOUT["target"], "weight"
    ].to_numpy().ravel()
    dD = self.D.copy()[
      :self.nodes, :self.nodes
    ]
    dD = adj2df(dD)
    dD = dD.loc[
      dD["source"] < dD["target"], "weight"
    ].to_numpy().ravel()
    # Create data ----
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    from scipy.stats import pearsonr
    ## Similarities ----
    r = pearsonr(dOUT, dIN)
    model.fit(dOUT.reshape(-1, 1), dIN)
    r2 = model.score(dOUT.reshape(-1, 1), dIN)
    self.stats = pd.DataFrame(
      {
        "X" : ["source"],
        "Y" : ["target"],
        "Cor" : [r[0]],
        "p-value": [r[1]],
        "R-squared" : [r2]
      }
    )
    ## t. similarity - distance ----
    r = pearsonr(dD, dIN)
    model.fit(dD.reshape(-1, 1), dIN)
    r2 = model.score(dD.reshape(-1, 1), dIN)
    self.stats = pd.concat(
      [
        self.stats,
        pd.DataFrame(
          {
            "X" : ["distance"],
            "Y" : ["target"],
            "Cor" : [r[0]],
            "p-value": [r[1]],
            "R-squared" : [r2]
          }
        )
      ]
    )
    
    ## s. similarity - distance ----
    r = pearsonr(dD, dOUT)
    model.fit(dD.reshape(-1, 1), dOUT)
    r2 = model.score(dD.reshape(-1, 1), dOUT)
    self.stats = pd.concat(
      [
        self.stats,
        pd.DataFrame(
          {
            "X" : ["distance"],
            "Y" : ["source"],
            "Cor" : [r[0]],
            "p-value": [r[1]],
            "R-squared" : [r2]
          }
        )
      ]
    )

  def get_freedman_diaconis_h(self, A, nlogo10=False):
    dA = adj2df(A.copy())
    dA = dA.loc[
      dA["weight"] != 0,
      "weight"
    ].to_numpy().reshape(-1)
    if nlogo10:
      dA = np.log10(dA)
    from scipy.stats import iqr
    n = dA.shape[0]
    return 2 * iqr(dA) * np.power(n, -1/3)

  def set_colregion(self, colregion : colregion):
    self.colregion = colregion

  def get_NOC_covers(self, overlap, Cr, labels, data_bar, dA):
    map_membership_id = membership2ids(Cr, dA)
    map_id_membership = invert_dict_multiple(map_membership_id)
    NOCS = {labels[x] : [] for x in overlap}
    for x in overlap:
      dO = data_bar.loc[
        (data_bar.nodes == labels[x]) &
        (data_bar.size > 0)
      ]
      dO = np.unique(dO.ids.to_numpy())
      for i in dO:
        if i in map_id_membership.keys():
          NOCS[labels[x]] = NOCS[labels[x]] + map_id_membership[i]
    for key in NOCS:
      NOCS[key] = list(np.unique(NOCS[key]))
    return NOCS

  def discover_overlap_nodes(self, K : int, Cr, labels):
    nocs = {}
    overlap = np.zeros(self.nodes)
    from scipy.cluster.hierarchy import cut_tree
    #######################
    # Without -1 ----
    dA = self.dA.copy()
    ## Cut tree ----
    dA["id"] = cut_tree(
      self.H, n_clusters=K
    ).reshape(-1)
    self.minus_one_Dc(dA)
    aesthetic_ids(dA)
    dA_ = dA.loc[dA["id"] != -1].copy()
    ## linkcom ids from 0 to k-1 ----
    dA_.id.loc[dA_.id > 0] = dA_.id.loc[dA_.id > 0] - 1
    ## Get lc sizes for each node ----
    data = bar_data(
      dA_, self.nodes, labels, norm=True
    )
    ## Get stats ----
    mean_s, std_s = stats_maxsize_linkcom(data)
    ## Grouped data ----
    size =  data.groupby("nodes")["size"].max()
    x = size < mean_s - 2 * std_s
    x = [
      np.where(labels == nd)[0][0] for i, nd in enumerate(x.index) if x.iloc[i]
    ]
    if len(x) > 0:
      nocs = self.get_NOC_covers(
        x, Cr, labels, data, dA_
      )
    overlap[x] += 1
    #######################
    # Max -1 ----
    ## Get lc sizes for each node ----
    data = bar_data(dA, self.nodes, labels, norm=True)
    ## Geta stats ----
    tree_nodes = tree_dominant_nodes(data, labels)
    if len(tree_nodes) > 0:
      for x in tree_nodes:
        if labels[x] in nocs.keys(): print("\n Warning: type I & II nocs collide\n")
        else: nocs[labels[x]] = [-1]
      overlap[tree_nodes] += 1
    return np.where(overlap > 0)[0], nocs

  def set_overlap_discovery(self, K, Cr, score):
    labels = self.colregion.labels[:self.nodes]
    overlap, _ = self.discover_overlap_nodes(K, Cr, labels) ### check <-----
    subdata = pd.DataFrame(
      {
        "labels": labels[overlap],
        "score" : [score] * len(overlap)
      }
    )
    self.overlap = pd.concat(
      [self.overlap, subdata],
      ignore_index=True
    )

  def set_overlap_labels(self, labels, score):
    subdata = pd.DataFrame(
      {
        "labels": labels,
        "score" : [score] * len(labels)
      }
    )
    self.overlap = pd.concat(
      [self.overlap, subdata],
      ignore_index=True
    )

  def get_ocn_discovery(self, K : int, Cr):
    labels = self.colregion.labels[:self.nodes]
    overlap, noc_covers = self.discover_overlap_nodes(K, Cr, labels)
    if len(overlap) > 0:
      return np.array([labels[i] for i in overlap]), noc_covers
    else:
      return np.array([]), noc_covers

  def get_ocn(self, *args):
    labels = self.colregion.labels[:self.nodes]
    overlap = skim_partition(args[0])
    return np.array([labels[i] for i, l in enumerate(overlap) if l == -1])