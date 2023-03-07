# Standard libs ----
import numpy as np
import pandas as pd
#  Personal libs ----
from various.network_tools import *
from modules.colregion import colregion
from modules.simanalysis import Sim
from process_hclust import ph
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
    self.subfolder = G.subfolder
    self.analysis = G.analysis
    # Compute similarity matrix ----
    self.similarity_by_feature_cpp()
    # Compute distance matrix ----
    self.dist_mat = self.linksim_matrix
    self.dist_mat[self.dist_mat == 0] = np.nan
    self.dist_mat = 1 - self.dist_mat
    self.dist_mat[np.isnan(self.dist_mat)] = np.nanmax(self.dist_mat) + 1
    # Compute hierarchy ----
    self.H = self.get_hierarchy()
    # Network to edgelist of EC ----
    self.dA = adj2df(
      self.A.copy()[:self.nodes, :]
    )
    ## Take out zeros ----
    self.dA = self.dA.loc[self.dA.weight != 0]
    ## Stats ----
    self.stat_cor()
    # Overlaps ----
    self.overlap = pd.DataFrame()
    # Cover ---
    self.cover = {}
    # KR ----
    self.kr = pd.DataFrame()
    # Entropy ----
    self.entropy = []

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
  
  def H_features_cpp(self, linkage, alpha, beta, cut=False):
    # Get network dataframe ----
    dA =  self.dA.copy()
    # Run process_hclust_fast.cpp ----
    features = ph(
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
    features.vite()
    return features

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
        BH = self.H_features_cpp(linkage, alpha, beta, cut=self.cut)
        BH_K = np.array(BH.get_K())
        BH_H = np.array(BH.get_Height())
        BH_NEC = np.array(BH.get_NEC())
        BH_MU = np.array(BH.get_MU())
        BH_D = np.array(BH.get_D())
        BH_ntrees = np.array(BH.get_ntrees())
        BH_X = np.array(BH.get_X())
        BH_OrP = np.array(BH.get_OrP())
        BH_XM = np.array(BH.get_XM())
        self.BH.append(
          pd.DataFrame(
            {
              "alpha" : [alpha] * len(BH_K),
              "beta" : [np.round(beta, 4)] * len(BH_K),
              "K" : BH_K,
              "height" : BH_H,
              "NEC" : BH_NEC,
              "mu" : BH_MU,
              "D" : BH_D,
              "ntrees": BH_ntrees,
              "X" : BH_X,
              "m" : BH_OrP,
              "xm" : BH_XM
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
              "n" : [features.alpha.iloc[0]],
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

  def discover_overlap_nodes(self, K : int, Cr, labels, s=2.):
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
    minus_one_Dc(dA)
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
    x = size < mean_s - s * std_s
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
  
  def set_cover(self, cover, score):
    self.cover[score] = cover

  def get_ocn_discovery(self, K : int, Cr, s=2.):
    labels = self.colregion.labels[:self.nodes]
    overlap, noc_covers = self.discover_overlap_nodes(K, Cr, labels, s=s)
    if len(overlap) > 0:
      return np.array([labels[i] for i in overlap]), noc_covers
    else:
      return np.array([]), noc_covers

  def get_ocn(self, *args):
    labels = self.colregion.labels[:self.nodes]
    overlap = skim_partition(args[0])
    return np.array([labels[i] for i, l in enumerate(overlap) if l == -1])