# Standard libs ----
import numpy as np
import pandas as pd
#  Personal libs ----
from various.network_tools import *
from various.discovery_channel import *
from modules.sign.simanalysis import Sim
from process_hclust import ph
from la_arbre_a_merde import noeud_arbre
from h_entropy import h_entropy as HE

class Hierarchy(Sim):
  def __init__(
    self, G, A, R, D, nodes, linkage, mode, lookup=0,
    undirected=False, architecture="product-positive"
  ):
    # Initialize Sim ---
    super().__init__(
      nodes, A, R, D, mode,
      topology=G.topology, index=G.index,
      lookup=lookup, undirected=undirected,
      architecture=architecture
    )
    # Set parameters
    self.linkage = linkage
    self.cut = G.cut
    self.pickle_path = G.pickle_path
    self.plot_path = G.plot_path
    self.subfolder = G.subfolder
    self.discovery = G.discovery
    self.analysis = G.analysis
    # Compute similarity matrix ----
    self.similarity_by_feature_cpp()
    # Compute distance matrix ----
    self.dist_mat = 1 - self.linksim_matrix
    # print(self.dist_mat)
    # print(np.sum(self.linksim_matrix < 0 ) )
    # Compute hierarchy ----
    self.H = self.get_hierarchy()
    self.delete_linksim_matrix()
    # Network to edgelist of EC ----
    non_x, non_y = np.where(self.nonzero[:self.nodes, :self.nodes])
    if not undirected:
       self.dA = pd.DataFrame(
        {
          "source" : list(non_x),
          "target" : list(non_y),
          "weight" : list(A[non_x, non_y])
        }
      )
    else:
      self.dA = pd.DataFrame(
        {
          "source" : list(non_x) + list(non_y),
          "target" : list(non_y) + list(non_x),
          "weight" : list(A[non_x, non_y]) * 2
        }
      )
    # Overlaps ----
    self.overlap = pd.DataFrame()
    # Rlabels ----
    self.rlabels = {}
    # Cover ---
    self.cover = {"source" : {}, "target" : {}, "both" : {}}
    # KR ----
    self.kr = pd.DataFrame()
    # Entropy ----
    self.entropy = []

  def delete_linksim_matrix(self):
    self.linksim_matrix = 0
    
  def delete_dist_matrix(self):
    self.dist_mat = 0

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
    print("Compute link hierarchical agglomeration ----")
    from scipy.cluster.hierarchy import linkage
    # from scipy.spatial.distance import squareform
    # print(squareform(self.dist_mat))
    return linkage(self.dist_mat, self.linkage)

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
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      alpha,
      beta,
      self.undirected
    )
    # features.bene("long")
    features.vite()
    return features
  
  def H_features_cpp_no_mu(self, linkage, alpha, beta, cut=False):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      alpha,
      beta,
      self.undirected
    )
    # features.bene("long")
    features.vite()
    result = np.array(
      [
        features.get_K(), features.get_Height(),
        features.get_NEC(),
        features.get_D(), features.get_ntrees(),
        features.get_X(), features.get_OrP(), features.get_XM(),
        features.get_S()
      ]
    )
    return result
  
  def H_features_cpp_nodewise(self, linkage, alpha, beta, cut=False):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      alpha,
      beta,
      self.undirected
    )
    # features.bene("long")
    k_equivalence = []
    k_equivalence.append(self.equivalence[0, 0])
    k_equivalence.append(1)
    for i in np.arange(self.equivalence.shape[0]-1):
      if self.equivalence[i, 1] != self.equivalence[i+1, 1]:
        k_equivalence.append(self.equivalence[i, 0])
    old = self.equivalence[0, 1]
    for i in np.arange(1, self.equivalence.shape[0]):
      if self.equivalence[i, 1] < old:
        k_equivalence.append(self.equivalence[i, 0])
        old = self.equivalence[i, 1]
    from collections import Counter
    count_r = dict(Counter(self.equivalence[:, 1]))
    count_r = {k : count_r[k] for k in -np.sort(-np.array(list(count_r.keys())))}
    t = 0
    for v in count_r.values():
      k_equivalence.append(self.equivalence[t + int(v/2), 0])
      t += int(v/2)
    k_equivalence = np.array(k_equivalence)
    k_equivalence = -np.sort(-np.unique(k_equivalence))
    features.vite_nodewise(
      k_equivalence, self.H[self.leaves - k_equivalence - 2, 2], k_equivalence.shape[0]
    )
    result = np.array(
      [
        features.get_K(), features.get_Height(),
        features.get_NEC(),
        features.get_D(), features.get_ntrees(),
        features.get_X(), features.get_OrP(), features.get_XM(),
        features.get_S()
      ]
    )
    return result
  
  def H_features_parallel_no_mu(self, listargs):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      listargs[0],
      listargs[3],
      listargs[1],
      listargs[2],
      self.undirected
    )
    # features.bene("long")
    features.vite()
    result = np.array(
      [
        features.get_K(), features.get_Height(),
        features.get_NEC(),
        features.get_D(), features.get_ntrees(),
        features.get_X(), features.get_OrP(), features.get_XM(),
        features.get_S()
      ]
    )
    return result, listargs[1], listargs[2]
  
  def H_features_parallel_mu(self, listargs):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      listargs[0],
      listargs[3],
      listargs[1],
      listargs[2],
      self.undirected
    )
    # features.bene("long")
    features.vite_mu()
    result = np.array([features.get_MU()])
    return result, listargs[1], listargs[2]
  
  def BH_features_parallel(self):
    import multiprocessing as mp
    # from various.pickle4reducer import Pickle4Reducer
    # ctx = mp.get_context()
    # ctx.reducer = Pickle4Reducer()

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
    # Create parameter list
    paralist = []
    for alpha in self.alpha:
      for beta in self.beta:
        paralist.append([linkage, alpha, beta, self.cut])
    with mp.Pool(4) as p:
      process = p.map(self.H_features_parallel_mu, paralist)
    results_no_mu = self.H_features_cpp_no_mu(linkage, 0, 0, self.cut)
    for feature, alpha, beta in process:
      self.BH.append(
        pd.DataFrame(
          {
            "alpha" : [alpha] * feature.shape[1],
            "beta" : [np.round(beta, 4)] * feature.shape[1],
            "K" : results_no_mu[0, :],
            "height" : results_no_mu[1, :],
            "NEC" : results_no_mu[2, :],
            "mu" : feature[0, :],
            "D" : results_no_mu[3, :],
            "ntrees": results_no_mu[4, :],
            "X" : results_no_mu[5, :],
            "m" : results_no_mu[6, :],
            "xm" : results_no_mu[7, :],
            "S" : results_no_mu[8, :],
            "SD" : (results_no_mu[3, :] / np.nansum(results_no_mu[3, :])) *  (results_no_mu[8, :] / np.nansum(results_no_mu[8, :])) 
          }
        )
      )
  
  def BH_features_cpp_no_mu(self):
    print("Mu-free")
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
    results_no_mu = self.H_features_cpp_no_mu(linkage, 0, 0, self.cut)
    self.BH.append(
      pd.DataFrame(
        {
          "alpha" : np.zeros(results_no_mu.shape[1]),
          "beta" : np.zeros(results_no_mu.shape[1]),
          "K" : results_no_mu[0, :],
          "height" : results_no_mu[1, :],
          "NEC" : results_no_mu[2, :],
          "mu" : np.zeros(results_no_mu.shape[1]),
          "D" : results_no_mu[3, :],
          "ntrees": results_no_mu[4, :],
          "X" : results_no_mu[5, :],
          "m" : results_no_mu[6, :],
          "xm" : results_no_mu[7, :],
          "S" : results_no_mu[8, :],
          "SD" : (results_no_mu[3, :] / np.nansum(results_no_mu[3, :])) * (results_no_mu[8, :] / np.nansum(results_no_mu[8, :]))
        }
      )
    )

  def BH_features_cpp_nodewise(self):
    print("Mu-free nodewise")
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
    results_nodewise = self.H_features_cpp_nodewise(linkage, 0, 0, self.cut)
    self.BH.append(
      pd.DataFrame(
        {
          "alpha" : np.zeros(results_nodewise.shape[1]),
          "beta" : np.zeros(results_nodewise.shape[1]),
          "K" : results_nodewise[0, :],
          "height" : results_nodewise[1, :],
          "NEC" : results_nodewise[2, :],
          "mu" : np.zeros(results_nodewise.shape[1]),
          "D" : results_nodewise[3, :],
          "ntrees": results_nodewise[4, :],
          "X" : results_nodewise[5, :],
          "m" : results_nodewise[6, :],
          "xm" : results_nodewise[7, :],
          "S" : results_nodewise[8, :]
        }
      )
    )

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
        BH = np.array(
          [
            BH.get_K(), BH.get_Height(), BH.get_NEC(), BH.get_MU(), BH.get_D(),
            BH.get_ntrees(), BH.get_X(), BH.get_OrP(), BH.get_XM()
          ]
        )
        self.BH.append(
          pd.DataFrame(
            {
              "alpha" : [alpha] * BH.shape[1],
              "beta" : [np.round(beta, 4)] * BH.shape[1],
              "K" : BH[0, :],
              "height" : BH[1, :],
              "NEC" : BH[2, :],
              "mu" : BH[3, :],
              "D" : BH[4, :],
              "ntrees": BH[5, :],
              "X" : BH[6, :],
              "m" : BH[7, :],
              "xm" : BH[8, :]
            }
          )
        )    

  def link_entropy_cpp(self, dist : str, cut=False):
    from scipy.cluster.hierarchy import cut_tree
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # Run process_hclust_fast.cpp ----
    entropy = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      3,
      0.1,
      self.undirected
    )

    # link_matrix = np.zeros((self.leaves, self.leaves))
    # for i in np.arange(self.leaves - 1, 0, -1):
    #   link_matrix[i - 1, :] = cut_tree(self.H, i).ravel()
    # link_matrix[self.leaves - 1, :] = np.arange(self.leaves)
    # link_matrix = link_matrix.astype(int)
    # link_height = np.zeros(self.leaves)
    # link_height[1:] = self.H[:, 2]

    entropy.arbre(dist)
    max_level = entropy.get_max_level()

    ento = entropy.get_entropy_h()[(self.leaves - max_level-1):]
    ento_h = entropy.get_entropy_v()[(self.leaves - max_level-1):]
    self.link_entropy = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy = np.sum(self.link_entropy)
    self.link_entropy = self.link_entropy / total_entropy

    ento = entropy.get_entropy_h_H()[(self.leaves - max_level-1):]
    ento_h = entropy.get_entropy_v_H()[(self.leaves - max_level-1):]
    self.link_entropy_H = np.array(
       [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy_H = np.sum(self.link_entropy_H)
    self.link_entropy_H = self.link_entropy_H / total_entropy_H

    sh = np.nansum(self.link_entropy[0, :])
    sv = np.nansum(self.link_entropy[1, :])
    print(f"\n\tlink entropy :  Sh : {sh:.4f}, and Sv : {sv:.4f}\n")
    sh = np.nansum(self.link_entropy_H[0, :])
    sv = np.nansum(self.link_entropy_H[1, :])
    print(f"\n\tlink entropy H: Sh : {sh:.4f}, and Sv : {sv:.4f}\n")

  
  def node_entropy_cpp(self, dist : str, cut=False):
    from scipy.cluster.hierarchy import cut_tree
    # Run process_hclust_fast.cpp ----
    entropy = HE(self.Z, self.nodes)

    # link_matrix = np.zeros((self.leaves, self.leaves))
    # for i in np.arange(self.leaves - 1, 0, -1):
    #   link_matrix[i - 1, :] = cut_tree(self.H, i).ravel()
    # link_matrix[self.leaves - 1, :] = np.arange(self.leaves)
    # link_matrix = link_matrix.astype(int)
    # link_height = np.zeros(self.leaves)
    # link_height[1:] = self.H[:, 2]

    entropy.arbre(dist)
    max_level = entropy.get_max_level()

    ento = entropy.get_entropy_h()[(self.nodes - max_level-1):]
    ento_h = entropy.get_entropy_v()[(self.nodes - max_level-1):]
    self.node_entropy = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy = np.sum(self.node_entropy)
    self.node_entropy = self.node_entropy / total_entropy

    ento = entropy.get_entropy_h_H()[(self.nodes - max_level-1):]
    ento_h = entropy.get_entropy_v_H()[(self.nodes - max_level-1):]
    self.node_entropy_H = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy_H = np.sum(self.node_entropy_H)
    self.node_entropy_H = self.node_entropy_H / total_entropy_H
    
    sh = np.nansum(self.node_entropy[0, :])
    sv = np.nansum(self.node_entropy[1, :])
    print(f"\n\tNode entropy :  Sh : {sh:.4f}, and Sv : {sv:.4f}\n")
    sh = np.nansum(self.node_entropy_H[0, :])
    sv = np.nansum(self.node_entropy_H[1, :])
    print(f"\n\tNode entropy H: Sh : {sh:.4f}, and Sv : {sv:.4f}\n")


  def la_abre_a_merde_cpp(self, features, undirected=None, sp=25):
    print("Compute node hierarchy ----")
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
    if not undirected: undirected = self.undirected

    if isinstance(undirected, bool):
      if undirected: undirected = 1
      else: undirected = 0

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
      features.shape[0],
      sp,
      undirected
    )
    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)

  def la_abre_a_merde_cpp_no_feat(self, sp=25):
    print("Compute node hierarchy no feat ----")
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
    NH = noeud_arbre(
      self.dist_mat,
      dA["source"].to_numpy(),
      dA["target"].to_numpy(),
      np.arange(self.leaves - 1, 0, -1, dtype=int),
      self.H[:, 2].ravel(),
      np.array([1] * (self.leaves - 1)),
      self.nodes,
      self.leaves,
      linkage,
      self.leaves - 1,
      sp,
      self.undirected
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
    if np.unique(dOUT).shape[0] == 1 or np.unique(dIN).shape[0] == 1:
      r = [np.nan] *2
    else:
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
    if np.unique(dD).shape[0] == 1 or np.unique(dIN).shape[0] == 1:
      r = [np.nan] *2
    else:
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
    if np.unique(dD).shape[0] == 1 or np.unique(dOUT).shape[0] == 1:
      r = [np.nan] *2
    else:
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

  def set_colregion(self, colregion):
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

  def set_overlap_labels(self, labels, score, direction):
    subdata = pd.DataFrame(
      {
        "labels": labels,
        "score" : [score] * len(labels),
        "direction" : [direction] * len(labels)
      }
    )
    self.overlap = pd.concat(
      [self.overlap, subdata],
      ignore_index=True
    )
  
  def set_cover(self, cover, score, direction : str):
    self.cover[direction][score] = cover

  def set_rlabels(self, rlabels, score, direction):
    none = np.sum(rlabels == -1)
    labels = rlabels.copy()
    labels[labels == -1] = np.arange(np.max(labels) + 1, np.max(labels) + 1 + none)
    self.rlabels[direction] = {"labels" : labels, "score" : score}

  def get_ocn(self, *args):
    labels = self.colregion.labels[:self.nodes]
    overlap = skim_partition(args[0])
    return np.array([labels[i] for i, l in enumerate(overlap) if l == -1])