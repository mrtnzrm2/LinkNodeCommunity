import numpy as np
import pandas as pd
from various.network_tools import *

def discover_overlap_nodes_2(H, K : int, Cr, labels, **kwargs):
  if "rho" in kwargs.keys():
    rho = kwargs["rho"]
    rho_state = rho
  else: rho=1.1
  if "sig" in kwargs.keys(): sig = kwargs["sig"]
  else: sig=0.5
  if "fun" in kwargs.keys(): fun = kwargs["fun"]
  else: fun = lambda x: np.log(1 + x)
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  from scipy.cluster.hierarchy import cut_tree
  #######################
  # Without -1 ----
  dA = H.dA.copy()
  ## Cut tree ----
  dA["id"] = cut_tree(H.H, n_clusters=K).reshape(-1)
  minus_one_Dc(dA)
  aesthetic_ids(dA)
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  for sn in single_nodes:
    dsn_src = dA.loc[dA.source == sn]
    wt_src = np.nanmean(fun(dsn_src.weight))
    dsn_tgt = dA.loc[dA.target == sn]
    wt_tgt = np.nanmean(fun(dsn_tgt.weight))
    # if labels[sn] == "5":
    #       print("SOURCE", wt_src)
    for lc in np.unique(dsn_src.id):
      lc_nodes = dsn_src.loc[dsn_src.id == lc]
      wc = np.nanmean(fun(lc_nodes.weight))
      lc_nodes = lc_nodes.target
      lc_nodes = set([i for i in lc_nodes])
      for ii, nsc in enumerate(NSC):
        if isinstance(rho_state, str):
          if "char" in rho_state:
            x = float(rho_state.split("r")[1])
            rho = np.nanstd(fun(dsn_src.weight)) / x
            rho = (wt_src + rho) / wt_src
          if "nchar" in rho_state:
            x = float(rho_state.split("r")[1])
            rho = np.nanstd(fun(dsn_src.weight)) / x
            rho = (wt_src - rho) / wt_src
        # if labels[sn] == "5":
        #   print(labels[list(nsc[0])], len(nsc[0].intersection(lc_nodes))/ len(nsc[0]), wc, rho * wt_src)
        if len(nsc[0].intersection(lc_nodes)) >= len(nsc[0]) * sig and wc >= rho * wt_src:
          if labels[sn] not in nocs.keys(): nocs[labels[sn]] = [nsc[1]]
          else: nocs[labels[sn]].append(nsc[1])
          overlap[sn] += 1
    # if labels[sn] == "5":
    #       print("TARGET", wt_tgt)
    for lc in np.unique(dsn_tgt.id):
      lc_nodes = dsn_tgt.loc[dsn_tgt.id == lc]
      wc = np.nanmean(fun(lc_nodes.weight))
      lc_nodes = lc_nodes.source
      lc_nodes = set([i for i in lc_nodes])
      for ii, nsc in enumerate(NSC):
        if isinstance(rho_state, str):
          if "char" in rho_state:
            x = float(rho_state.split("r")[1])
            rho = np.nanstd(fun(dsn_tgt.weight)) / x
            rho = (wt_tgt + rho) / wt_tgt
          if "nchar" in rho_state:
            x = float(rho_state.split("r")[1])
            rho = np.nanstd(fun(dsn_tgt.weight)) / x
            rho = (wt_tgt - rho) / wt_tgt
        # if labels[sn] == "5":
        #   print(labels[list(nsc[0])], len(nsc[0].intersection(lc_nodes))/ len(nsc[0]), wc, rho * wt_tgt)
        if len(nsc[0].intersection(lc_nodes)) >= len(nsc[0]) * sig and wc >= rho * wt_tgt:
          if labels[sn] not in nocs.keys(): nocs[labels[sn]] = [nsc[1]]
          else: nocs[labels[sn]].append(nsc[1])
          overlap[sn] += 1
    for k in nocs.keys():
      nocs[k] = list(np.unique(nocs[k]))
    #######################
    # Max -1 ----
    # Get lc sizes for each node ----
    data = bar_data(dA, H.nodes, labels, norm=True)
    ## Geta stats ----
    tree_nodes = tree_dominant_nodes(data, labels)
    tree_nodes = [v for v in tree_nodes if skimCr[v] == -1]
    if len(tree_nodes) > 0:
      for x in tree_nodes:
        if labels[x] not in nocs.keys() and skimCr[x] == -1: nocs[labels[x]] = [-1]
      overlap[tree_nodes] += 1
    ## Check unassigned nodes ----
    for sn in single_nodes:
      if labels[sn] not in nocs.keys():
        nocs[labels[sn]] = [-1]
        overlap[sn] += 1
    return np.where(overlap > 0)[0], nocs

def discover_overlap_nodes_3(H, K : int, Cr, labels, undirected=False, **kwargs):
  from scipy.stats import skew
  from scipy.cluster.hierarchy import cut_tree
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  #######################
  # Without -1 ----
  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)
  minus_one_Dc(dA, undirected)
  aesthetic_ids(dA)
  # Sim matrix to Dist ---
  Dsource = H.source_sim_matrix
  Dsource[Dsource == 0] = np.nan
  Dsource = 1/Dsource + 1
  sk = np.abs(skew(Dsource[np.isnan(Dsource)]))
  if not np.isnan(sk) and sk > 1:
    Dsource[np.isnan(Dsource)] = np.nanmax(Dsource) + np.nanstd(Dsource) * sk
  else: 
    Dsource[np.isnan(Dsource)] = np.nanmax(Dsource) + np.nanstd(Dsource)
  np.fill_diagonal(Dsource, np.nan)
  Dtarget = H.target_sim_matrix
  Dtarget[Dtarget == 0] = np.nan
  Dtarget = 1/Dtarget + 1
  sk = np.abs(skew(Dtarget[np.isnan(Dtarget)]))
  if not np.isnan(sk) and sk > 1:
    Dtarget[np.isnan(Dtarget)] = np.nanmax(Dtarget) + np.nanstd(Dtarget) * sk
  else: 
    Dtarget[np.isnan(Dtarget)] = np.nanmax(Dtarget) + np.nanstd(Dtarget)
  np.fill_diagonal(Dtarget, np.nan)
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  # print(NSC)
  for sn in single_nodes:
    dsn_src = dA.loc[dA.source == sn]
    dsn_tgt = dA.loc[dA.target == sn]
    Dsn = np.zeros((len(NSC), 4)) * np.nan
    # if sn == 24:
    #   print(dsn_src)
    #   print(dsn_tgt)
    for ii, nsc in enumerate(NSC):
      neighbor_nodes_src = list(set(dsn_src.target).intersection(nsc[0]))
      neighbor_nodes_tgt = list(set(dsn_tgt.source).intersection(nsc[0]))
      # if sn == 24:
      #   print(neighbor_nodes_tgt)
        # print(neighbor_nodes_src)
      if len(neighbor_nodes_src) > 0:
        Dsn[ii, 0] = np.nanmean(Dsource[sn, neighbor_nodes_src])
        Dsn[ii, 1] = np.nanmean(Dtarget[sn, neighbor_nodes_src])
      if len(neighbor_nodes_tgt) > 0:
        Dsn[ii, 2] = np.nanmean(Dtarget[sn, neighbor_nodes_tgt])
        Dsn[ii, 3] = np.nanmean(Dsource[sn, neighbor_nodes_tgt])
    Dsn = np.nanmean(Dsn, axis=1)
    # if sn == 24:
    #   print(Dsn)
    # Dsn[np.isnan(Dsn)] = np.sqrt(2) * np.nanmax(Dsn)
    # if sn == 24:
    #   print(Dsn)
    dsn = np.exp(np.nanmean(np.log(Dsn)))
    scale = np.exp(np.nanstd(np.log(Dsn)))
    rev = np.abs(np.exp(skew(np.log(Dsn[~np.isnan(Dsn)]))))
    if not np.isnan(rev) and rev < 1/2:
      dsn = dsn + rev * scale
    else:
      dsn = dsn + scale / 2
    # dsn = np.nanmean(Dsn)
    if ~np.isnan(dsn):
      for ii, nsc in enumerate(NSC):
        if ~np.isnan(Dsn[ii]) :
          if Dsn[ii] < dsn:
            if labels[sn] not in nocs.keys(): nocs[labels[sn]] = [nsc[1]]
            else: nocs[labels[sn]].append(nsc[1])
            overlap[sn] += 1
  for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      # nocs[labels[sn]] = [-1]
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs

def discover_overlap_nodes_5(H, K : int, Cr, labels, undirected=False, **kwargs):  
  from itertools import combinations
  from scipy.cluster.hierarchy import cut_tree
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  #######################
  # Without -1 ----
  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)
  minus_one_Dc(dA, undirected)
  aesthetic_ids(dA)
  Dsource = 1 / H.source_sim_matrix + 1
  Dtarget = 1 / H.target_sim_matrix + 1
  m = np.maximum(np.max(Dsource[Dsource < np.Inf]), np.max(Dtarget[Dtarget < np.Inf]))
  Dsource[Dsource == np.Inf] = np.max(Dsource[Dsource < np.Inf])
  Dtarget[Dtarget == np.Inf] = np.max(Dtarget[Dtarget < np.Inf])
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  for sn in single_nodes:
    dsn_src = dA.loc[dA.source == sn]
    dsn_tgt = dA.loc[dA.target == sn]
    Dsn = np.zeros((len(NSC), 2))
    for ii, nsc in enumerate(NSC):
      neighbor_nodes_src = set(dsn_src.target).intersection(nsc[0])
      neighbor_nodes_tgt = set(dsn_tgt.source).intersection(nsc[0])
      neighbors = list(neighbor_nodes_src.union(neighbor_nodes_tgt))
      if len(neighbors) > 0:
        Dsn[ii, 0] = np.nanmean(Dsource[sn, neighbors])
        Dsn[ii, 1] = np.nanmean(Dtarget[sn, neighbors])
    if len(NSC) > 1:
      comb = [(u, v) for u, v in combinations(range(len(NSC)), 2)]
    else: comb = [(0, 0)]
    Dsn[Dsn == 0] = np.nan
    Rc = np.nanmean(Dsn, axis=0)
    Dsn[np.isnan(Dsn)] = m
    Rs = np.array([np.linalg.norm(Dsn[u] - Dsn[v]) for u, v in comb])
    Rs = np.argmin(Rs)
    Rs = np.abs(Dsn[comb[Rs][0]] - Dsn[comb[Rs][1]])
    Rc = Rc + Rs / (len([i for i in range(len(NSC)) if np.linalg.norm(Dsn[i]) < np.sqrt(2 * np.power(m,2))]))
    Rc = np.linalg.norm(Rc)
    Dsn = np.linalg.norm(Dsn, axis=1)
    for ii, nsc in enumerate(NSC):
      if not np.isnan(Dsn[ii]):
        if Dsn[ii] <= Rc and Dsn[ii] < np.sqrt(2 * np.power(m,2)):
          if labels[sn] not in nocs.keys(): nocs[labels[sn]] = [nsc[1]]
          else: nocs[labels[sn]].append(nsc[1])
          overlap[sn] += 1
  for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs

def discover_overlap_nodes_6(H, K : int, Cr, labels, undirected=False, **kwargs):  
  from itertools import combinations
  from scipy.cluster.hierarchy import cut_tree
  nocs_size = {}
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)
  minus_one_Dc(dA, undirected)
  aesthetic_ids(dA)
  Dsource = 1 / H.source_sim_matrix + 1
  Dtarget = 1 / H.target_sim_matrix + 1
  Dsource[Dsource == np.Inf] = np.max(Dsource[Dsource < np.Inf])
  Dtarget[Dtarget == np.Inf] = np.max(Dtarget[Dtarget < np.Inf])
  m_s = np.nanmax(Dsource)
  m_t = np.nanmax(Dtarget)
  max_m = np.sqrt(np.power(m_s, 2) + np.power(m_t, 2))
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  for sn in single_nodes:
    dsn_src = dA.loc[dA.source == sn]
    dsn_tgt = dA.loc[dA.target == sn]
    Dsn = np.zeros((len(NSC), 2))
    for ii, nsc in enumerate(NSC):
      neighbor_nodes_src = set(dsn_src.target).intersection(nsc[0])
      neighbor_nodes_tgt = set(dsn_tgt.source).intersection(nsc[0])
      neighbors = list(neighbor_nodes_src.union(neighbor_nodes_tgt))
      if len(neighbors) > 0:
        Dsn[ii, 0] = np.nanmin(Dsource[sn, neighbors])
        Dsn[ii, 1] = np.nanmin(Dtarget[sn, neighbors])
    if len(NSC) > 1:
      comb = [(u, v) for u, v in combinations(range(len(NSC)), 2)]
    else: comb = [(0, 0)]
    Dsn[Dsn == 0] = np.nan
    Rc = np.nanmean(Dsn, axis=0)
    Dsn[np.isnan(Dsn[:, 0]), 0] = m_s
    Dsn[np.isnan(Dsn[:, 1]), 1] = m_t
    # if labels[sn] == "teo":
    #   print(Rc)
    Rs = np.array([np.linalg.norm(Dsn[u] - Dsn[v]) for u, v in comb if np.linalg.norm(Dsn[u]) < max_m and np.linalg.norm(Dsn[v]) < max_m])
    if len(Rs) > 0:
      Rs = np.argmin(Rs)
    else: Rs = 0
    Rs = np.abs(Dsn[comb[Rs][0]] - Dsn[comb[Rs][1]])
    Rc = Rc + Rs / (len([i for i in range(len(NSC)) if np.linalg.norm(Dsn[i]) < max_m]))
    # if labels[sn] == "teo":
    #   print(Rs)
    #   print(Rc)
    #   print(Dsn)
    Rc = np.linalg.norm(Rc)
    Dsn = np.linalg.norm(Dsn, axis=1)
    for ii, nsc in enumerate(NSC):
      if not np.isnan(Dsn[ii]):
        if Dsn[ii] <= Rc and Dsn[ii] < max_m:
          if labels[sn] not in nocs.keys():
            nocs[labels[sn]] = [nsc[1]]
            nocs_size[labels[sn]] = {nsc[1] : np.exp(-Dsn[ii]/max_m)}
          else:
            nocs[labels[sn]].append(nsc[1])
            nocs_size[labels[sn]].update({nsc[1] : np.exp(-Dsn[ii]/max_m)})
          overlap[sn] += 1
  # for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs, nocs_size

def discover_overlap_nodes_4(H, K : int, Cr, labels, undirected=False, **kwargs):
  from scipy.stats import skew
  from scipy.cluster.hierarchy import cut_tree
  nocs_size = {}
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  #######################
  # Without -1 ----
  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)
  minus_one_Dc(dA, undirected)
  aesthetic_ids(dA)
  # Sim matrix to Dist ---
  Dsource = H.source_sim_matrix
  Dsource[Dsource == 0] = np.nan
  Dsource = 1/Dsource + 1
  sk = np.abs(skew(Dsource[np.isnan(Dsource)]))
  if not np.isnan(sk) and sk > 1:
    Dsource[np.isnan(Dsource)] = np.nanmax(Dsource) + np.nanstd(Dsource) * sk
  else: 
    Dsource[np.isnan(Dsource)] = np.nanmax(Dsource) + np.nanstd(Dsource)
  np.fill_diagonal(Dsource, np.nan)
  Dtarget = H.target_sim_matrix
  Dtarget[Dtarget == 0] = np.nan
  Dtarget = 1/Dtarget + 1
  sk = np.abs(skew(Dtarget[np.isnan(Dtarget)]))
  if not np.isnan(sk) and sk > 1:
    Dtarget[np.isnan(Dtarget)] = np.nanmax(Dtarget) + np.nanstd(Dtarget) * sk
  else: 
    Dtarget[np.isnan(Dtarget)] = np.nanmax(Dtarget) + np.nanstd(Dtarget)
  np.fill_diagonal(Dtarget, np.nan)
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  # print(NSC)
  for sn in single_nodes:
    dsn_src = dA.loc[dA.source == sn]
    dsn_tgt = dA.loc[dA.target == sn]
    Dsn_src = np.zeros(len(NSC)) * np.nan
    Dsn_tgt = np.zeros(len(NSC)) * np.nan
    for ii, nsc in enumerate(NSC):
      neighbor_nodes_src = list(set(dsn_src.target).intersection(nsc[0]))
      neighbor_nodes_tgt = list(set(dsn_tgt.source).intersection(nsc[0]))
      if len(neighbor_nodes_src) > 0:
        Dsn_src[ii] = np.nanmean(Dsource[sn, neighbor_nodes_src])
        if Dsn_src[ii] == np.Inf: Dsn_src[ii] = np.nan
      if len(neighbor_nodes_tgt) > 0:
        Dsn_tgt[ii] = np.nanmean(Dtarget[sn, neighbor_nodes_tgt])
        if Dsn_tgt[ii] == np.Inf: Dsn_tgt[ii] = np.nan
    dsn_src = np.exp(np.nanmean(np.log(Dsn_src)))
    dsn_tgt = np.exp(np.nanmean(np.log(Dsn_tgt)))

    scale_src = np.exp(np.nanstd(np.log(Dsn_src)))
    rev_src = np.abs(np.exp(skew(np.log(Dsn_src[~np.isnan(Dsn_src)]))))
    if not np.isnan(rev_src) and rev_src < 1/2:
      dsn_src = dsn_src + rev_src * scale_src
    else:
      dsn_src = dsn_src + scale_src / 2

    scale_tgt = np.exp(np.nanstd(np.log(Dsn_tgt)))
    rev_tgt = np.abs(np.exp(skew(np.log(Dsn_tgt[~np.isnan(Dsn_tgt)]))))
    if not np.isnan(rev_tgt) and rev_tgt < 1/2:
      dsn_tgt = dsn_tgt + rev_tgt * scale_tgt
    else:
      dsn_tgt = dsn_tgt + scale_tgt / 2

    for ii, nsc in enumerate(NSC):
      if not np.isnan(Dsn_src[ii]) and not np.isnan(Dsn_tgt[ii]):
        if Dsn_src[ii] < dsn_src and Dsn_tgt[ii] < dsn_tgt:
          if labels[sn] not in nocs.keys():
            nocs[labels[sn]] = [nsc[1]]
            nocs_size[labels[sn]] = {nsc[1] : np.exp(-Dsn_src[ii])}
          else:
            nocs[labels[sn]].append(nsc[1])
            nocs_size[labels[sn]].update({nsc[1] : np.exp(-Dsn_src[ii])})
          overlap[sn] += 1
      else:
        if not np.isnan(dsn_src):
          if not np.isnan(Dsn_src[ii]):
            if Dsn_src[ii] < dsn_src:
              if labels[sn] not in nocs.keys():
                nocs[labels[sn]] = [nsc[1]]
                nocs_size[labels[sn]] = {nsc[1] : np.exp(-Dsn_src[ii])}
              else:
                nocs[labels[sn]].append(nsc[1])
                nocs_size[labels[sn]].update({nsc[1] : np.exp(-Dsn_src[ii])})
              overlap[sn] += 1
        if not np.isnan(dsn_tgt):
          if not np.isnan(Dsn_tgt[ii]):
            if Dsn_tgt[ii] < dsn_tgt:
              if labels[sn] not in nocs.keys():
                nocs[labels[sn]] = [nsc[1]]
                nocs_size[labels[sn]] = {nsc[1] : np.exp(-Dsn_tgt[ii])}
              else:
                nocs[labels[sn]].append(nsc[1])
                nocs_size[labels[sn]].update({nsc[1] : np.exp(-Dsn_tgt[ii])})
              overlap[sn] += 1
  for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      # nocs[labels[sn]] = [-1]
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs, nocs_size

def discover_overlap_nodes(H, K : int, Cr, labels, s=2.):
  nocs = {}
  overlap = np.zeros(H.nodes)
  from scipy.cluster.hierarchy import cut_tree
  #######################
  # Without -1 ----
  dA = H.dA.copy()
  ## Cut tree ----
  dA["id"] = cut_tree(
    H.H, n_clusters=K
  ).reshape(-1)
  minus_one_Dc(dA)
  aesthetic_ids(dA)
  dA_ = dA.loc[dA["id"] != -1].copy()
  ## linkcom ids from 0 to k-1 ----
  dA_.id.loc[dA_.id > 0] = dA_.id.loc[dA_.id > 0] - 1
  ## Get lc sizes for each node ----
  data = bar_data(
    dA_, H.nodes, labels, norm=True
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
    nocs = H.get_NOC_covers(
      x, Cr, labels, data, dA_
    )
  overlap[x] += 1
  #######################
  # Max -1 ----
  ## Get lc sizes for each node ----
  data = bar_data(dA, H.nodes, labels, norm=True)
  ## Geta stats ----
  tree_nodes = tree_dominant_nodes(data, labels)
  if len(tree_nodes) > 0:
    for x in tree_nodes:
      if labels[x] in nocs.keys(): print("\n Warning: type I & II nocs collide\n")
      else: nocs[labels[x]] = [-1]
    overlap[tree_nodes] += 1
  return np.where(overlap > 0)[0], nocs

def get_ocn_discovery(H, K : int, Cr, s=2.):
    labels = H.colregion.labels[:H.nodes]
    overlap, noc_covers = discover_overlap_nodes(H, K, Cr, labels, s=s)
    if len(overlap) > 0:
      return np.array([labels[i] for i in overlap]), noc_covers
    else:
      return np.array([]), noc_covers
    
def discovery_2(H, K : int, Cr, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_2(H, K, Cr, labels, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_3(H, K : int, Cr, undirected=False, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_3(H, K, Cr, labels, undirected=undirected, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_4(H, K : int, Cr, undirected=False, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, sizes = discover_overlap_nodes_4(H, K, Cr, labels, undirected=undirected, **kwargs)
  return np.array([labels[i] for i in overlap]), noc_covers, sizes
  
def discovery_5(H, K : int, Cr, undirected=False, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_5(H, K, Cr, labels, undirected=undirected, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_6(H, K : int, Cr, undirected=False, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes = discover_overlap_nodes_6(H, K, Cr, labels, undirected=undirected, **kwargs)
  return np.array([labels[i] for i in overlap]), noc_covers, noc_sizes