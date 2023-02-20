# General libraries ----
from copy import copy
import numpy as np
import pandas as pd
import os
# My libraries ----
from various.network_tools import *
from modules.colregion import colregion
from modules.simanalysis import Sim

# Initiliaze process_hclust c++
import process_hclust as ph
from la_arbre_a_merde import noeud_arbre

class Node:
  ### Class that helps in the la abre de merde ###
  def __init__(self, tip, nodes):
    self.tip = tip
    self.nodes = nodes

class Hierarchy(Sim):
  def __init__(
    self, G, A, D, nodes, linkage, mode,
    prob=False, sln=False, **kwargs
  ):
    # Initialize Sim ---
    super().__init__(
      nodes, A.copy(), D.copy(), mode,
      nlog10=G.nlog10, prob=prob, lookup=G.lookup,
      mapping=G.mapping, topology=G.topology, index=G.index, **kwargs
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
    self.similarity_by_feature()
    # Compute distance matrix ----
    self.dist_mat = self.sim_mat.copy()
    self.dist_mat[self.dist_mat == 0] = np.nan
    self.dist_mat = 1 - self.dist_mat
    self.dist_mat[np.isnan(self.dist_mat)] = np.nanmax(self.dist_mat) + 100
    # Compute hierarchy ----
    self.H = self.get_hierarchy()
    # Network to edgelist of EC ----
    self.dA = adj2df(
      self.A.copy()[:self.nodes, :]
    )
    if sln:
      dSLN = adj2df(
        self.sln.copy()[:self.nodes, :]
      )
      self.dA["sln"] = dSLN.weight
    self.sln_h_parameters = {}
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

  #** Single-linkage code:

  def which_tip(self, tree, nodes):
    ntip = len(tree)
    val = [False] * ntip
    for i, t in enumerate(tree):
      if len(np.intersect1d(t.nodes, nodes)) >= 1:
        val[i] = True
    return val

  def knot(self, tree_i, tree_f, ind_tips, h):
    print(tree_i[ind_tips[0]].nodes)
    print(tree_i[ind_tips[1]].nodes)
    print("________")
    ### Combine tips ----
    combine_nodes = np.sort(
      np.unique(
        np.concatenate(
          (
            tree_i[ind_tips[0]].nodes,
            tree_i[ind_tips[1]].nodes
          ),
          axis=0
        )
      )
    )
    ### Assign to Z ----
    self.Z[self.t, 0] = tree_i[ind_tips[0]].tip
    self.Z[self.t, 1] = tree_i[ind_tips[1]].tip
    self.Z[self.t, 2] = h
    self.Z[self.t, 3] = len(combine_nodes)
    ## Add new tip ----
    tree_f.append(
      Node(
        self.nodes + self.t,
        combine_nodes
      )
    )
    ## Eliminate used tips ----
    for j, tip in enumerate(ind_tips):
      del tree_i[tip - j]
    self.t += 1

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

  def la_abre_a_merde(self, features):
    from scipy.cluster.hierarchy import cut_tree
    # Get features and filter over NEC >= 1 ---
    NEC = np.where(features["NEC"] >= 1)[0]
    K = features["K"].iloc[NEC]
    H = features["height"].iloc[NEC]
    # Elliminate H duplicates in H and K ----
    H, dpl = np.unique(H, return_index=True)
    K = K[dpl]
    # Define the nodes ----
    tree = []
    for i in np.arange(self.nodes):
      tree.append(
        Node(i, np.array([i]))
      )
    # Create Z ----
    self.Z = np.zeros((self.nodes - 1, 4))
    # Create network dataframe ----
    dA = self.dA.copy()
    # Start loop over partitions ----
    self.t = 0
    for k, h in zip(K, H):
      ## Cut tree ----
      leaves_ids = cut_tree(
        self.H,
        n_clusters = k
      ).reshape(-1)
      dA["id"] =  leaves_ids
      ## Assign tree LC memberships to -1 ----
      self.minus_one_Dc(dA)
      ## Get effective LC memberships ----
      ids = dA.query("id >= 0")["id"].to_numpy()
      ids = np.sort(np.unique(ids))
      ## Copy tree ---
      tree_c = copy(tree)
      ## Scan partition ----
      for id in ids:
        ### Filter id ----
        dAid = dA.loc[dA["id"] == id]
        ### Get core nodes ----
        src = dAid["source"].to_numpy()
        tgt = dAid["target"].to_numpy()
        inter = np.unique(
          np.intersect1d(src, tgt)
        )
        if len(inter) > 1:
          ### Search for compatible parents ----
          compatible_tip = np.where(
            self.which_tip(tree_c, inter)
          )[0]
          if len(compatible_tip) == 2:
            self.knot(
              tree_c, tree_c,
              compatible_tip,
              h
            )
          elif len(compatible_tip) > 2:
            ### Copy the compatible tips ----
            tree_c2 = []
            for tip in compatible_tip:
              tree_c2.append(
                tree_c[tip]
              )
            ### Eliminate compatible tips from original ----
            for j, tip in enumerate(compatible_tip):
              del tree_c[tip - j]
            while len(tree_c2) >= 2:
              if len(tree_c2) > 2:
                self.knot(
                  tree_c2, tree_c2,
                  np.array([0, 1]),
                  h
                )
              elif len(tree_c2) == 2:
                self.knot(
                  tree_c2, tree_c,
                  np.array([0, 1]),
                  h
                )
          tree = tree_c
  
  def stat_cor(self):
    # Copy target and source similarity matrices ----
    dIN = self.sim1.copy()
    dIN[np.isnan(dIN)] = np.nanmin(dIN) - 1
    dIN = adj2df(dIN)
    dIN = dIN.loc[
      dIN["source"] < dIN["target"], "weight"
    ].to_numpy().ravel()
    dOUT = self.sim2.copy()
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

  ##########################################################################################

  ## SLN methods ----

  def get_parameter_stan(self, fit : pd.DataFrame, param, shape):
    mean_shape = np.zeros(shape)
    std_shape = np.zeros(shape)
    name_shape = np.zeros(shape)
    name_shape = name_shape + 120022.393
    name_shape = name_shape.astype(str)
    col_names = fit.columns.to_numpy()
    extracted_names = []
    extracted_indices = []
    for col in col_names:
      if param in col:
        extracted_names.append(col)
        s = col.split(".")
        if len(s) > 1:
          s = s[1:]
          s = np.array([int(ii) for ii in s])
        else: s = np.array([0])
        extracted_indices.append(s)
    mean_parameters = fit[extracted_names].mean().to_numpy()
    std_parameters = fit[extracted_names].std().to_numpy()
    for i, idx in enumerate(extracted_indices):
      idx = idx - 1
      if len(idx) == 1:
        mean_shape[idx[0]] = mean_parameters[i]
        std_shape[idx[0]] = std_parameters[i]
        name_shape[idx[0]] = extracted_names[i]
      elif len(idx) == 2:
        mean_shape[idx[0], idx[1]] = mean_parameters[i]
        std_shape[idx[0], idx[1]] = std_parameters[i]
        name_shape[idx[0], idx[1]] = extracted_names[i]
      elif len(idx) == 3:
        mean_shape[idx[0], idx[1], idx[2]] = mean_parameters[i]
        std_shape[idx[0], idx[1], idx[2]] = std_parameters[i]
        name_shape[idx[0], idx[1], idx[2]] = extracted_names[i]
    return name_shape, mean_shape, std_shape

  def sln_hierarchy_stan(self, supra, infra, labels):
    sup = supra.copy().astype(np.double)
    inf = infra.copy().astype(np.double)
    nolinks = self.A == 0
    sup[nolinks] = np.nan
    inf[nolinks] = np.nan
    ##
    sup = adj2df(sup)
    inf = adj2df(inf)
    sup = sup.loc[~np.isnan(sup.weight)]
    inf = inf.loc[~np.isnan(inf.weight)]
    ##
    target_index = sup.target.to_numpy().astype(int)
    source_index = sup.source.to_numpy().astype(int)
    leaves = len(target_index)
    v1c = np.where(labels == "v1c")[0].astype(int)
    ##
    data = {
      "inj" : self.A.shape[1],
      "n" : self.A.shape[0],
      "l" : leaves,
      "supra" : sup.weight.to_numpy().astype(int),
      "infra" : inf.weight.to_numpy().astype(int),
      "tgt" : target_index + 1,
      "src" : source_index + 1,
      "v1c" : v1c[0] + 1
    }
    ##
    import stan
    from STAN_models.SLN.sln_hierarchy_org import beta_binomial
    posterior = stan.build(beta_binomial, data=data)
    fit_frame = posterior.sample(
      num_chains=4, num_warmup=1000, num_samples=1000
    ).to_frame()
    ## Get parameters from stan ----
    par_name = ["beta", "sig"]
    par_shape = [(self.A.shape[0]), (1)]
    for par, pshape in zip(par_name, par_shape):
      para_names, para_mu, para_std = self.get_parameter_stan(fit_frame, par, pshape)
      self.sln_h_parameters[par] = (para_names, para_mu, para_std)