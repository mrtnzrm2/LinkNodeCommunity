import numpy as np
import seaborn as sns
import numpy.typing as npt
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
  np.seterr(divide='ignore', invalid='ignore')
  Dsource = 1 / H.source_sim_matrix - 1
  np.seterr(divide='ignore', invalid='ignore')
  Dtarget = 1 / H.target_sim_matrix - 1
  Dsource[Dsource == np.Inf] = np.max(Dsource[Dsource < np.Inf]) * 1.05
  Dtarget[Dtarget == np.Inf] = np.max(Dtarget[Dtarget < np.Inf]) * 1.05

  Dsource = np.log(Dsource)
  Dtarget = np.log(Dtarget)

  Dsource += np.abs(np.min(Dsource))
  Dtarget += np.abs(np.min(Dtarget))

  m_s = np.nanmax(Dsource)
  m_t = np.nanmax(Dtarget)
  max_D = np.sqrt(np.power(m_s, 2) + np.power(m_t, 2))

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
        Dsn[ii, 0] = np.nanmin(Dsource[sn, neighbors]) / max_D
        Dsn[ii, 1] = np.nanmin(Dtarget[sn, neighbors]) / max_D
      else:
        Dsn[ii, 0] = m_s / max_D
        Dsn[ii, 1] = m_t / max_D

    if len(NSC) > 1:
      comb = [(u, v) for u, v in combinations(range(len(NSC)), 2)]
    else: comb = [(0, 0)]

    Rc = np.nanmean(Dsn, axis=0)

    # if labels[sn] == "teo":
    #   print(Rc)
    Rs = np.array([np.linalg.norm(Dsn[u] - Dsn[v]) for u, v in comb if np.linalg.norm(Dsn[u]) < 1 and np.linalg.norm(Dsn[v]) < 1])
    if len(Rs) > 0:
      Rs = np.argmin(Rs)
    else: Rs = 0
    Rs = np.abs(Dsn[comb[Rs][0]] - Dsn[comb[Rs][1]])
    Rc = Rc + Rs / (len([i for i in range(len(NSC)) if np.linalg.norm(Dsn[i]) < 1]))
    # if labels[sn] == "teo":
    #   print(Rs)
    #   print(Rc)
    #   print(Dsn)
    Rc = np.linalg.norm(Rc)
    Dsn = np.linalg.norm(Dsn, axis=1)
    for ii, nsc in enumerate(NSC):
      if not np.isnan(Dsn[ii]):
        if Dsn[ii] <= Rc and Dsn[ii] < 1:
          if labels[sn] not in nocs.keys():
            nocs[labels[sn]] = [nsc[1]]
            nocs_size[labels[sn]] = {nsc[1] : 1 - Dsn[ii]}
          else:
            nocs[labels[sn]].append(nsc[1])
            nocs_size[labels[sn]].update({nsc[1] : 1 - Dsn[ii]})
          overlap[sn] += 1
  # for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs, nocs_size

def get_nocs_information(H, Cr, dA, labels, direction, index):
  
  from scipy.cluster.hierarchy import linkage, cut_tree
  from scipy.spatial.distance import squareform

  nocs_size = {}
  nocs = {}
  overlap = np.zeros(H.nodes)

  if index == "D1_2_4" or index == "dist_sim":
    if direction == "source":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.source_sim_matrix) - 1
      Diss[Diss == np.Inf] = np.max(Diss[Diss < np.Inf]) * 1.05
    elif direction == "target":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.target_sim_matrix) - 1
      Diss[Diss == np.Inf] = np.max(Diss[Diss < np.Inf]) * 1.05
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "Hellinger2":
    if direction == "source":
      Diss = np.sqrt(1 - H.source_sim_matrix)
    elif direction == "target":
      Diss = np.sqrt(1 - H.target_sim_matrix)
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "cos":
    if direction == "source":
      Diss = 1 - H.source_sim_matrix
    elif direction == "target":
      Diss = 1 - H.target_sim_matrix
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "Shortest_Path":
    if direction == "source":
      Diss = H.source_sim_matrix
    elif direction == "target":
      Diss = H.target_sim_matrix
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "bsim_2" or index == "bsim":
    if direction == "source":
      Diss = H.source_sim_matrix
      mindss = np.nanmin(Diss)
      Diss = Diss + mindss
      Diss = Diss / np.nanmax(Diss)
      Diss = 1 - Diss
    elif direction == "target":
      Diss = 1 - H.target_sim_matrix
      mindss = np.nanmin(Diss)
      Diss = Diss + mindss
      Diss = Diss / np.nanmax(Diss)
      Diss = 1 - Diss
    else:
      raise ValueError("No accepted direction in discovery channel")
  else: raise ValueError("No accepeted index in discovery channel")

  th = np.max(Diss) * 1.05
  
  ## Single nodes ----
  single_nodes = np.where(Cr == -1)[0]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(Cr == i)[0]), i) for i in np.unique(Cr) if i != -1]

  for sn in single_nodes:
    if direction == "source":
      dsn = set(dA.loc[dA.source == sn].target)
    elif direction == "target":
      dsn = set(dA.loc[dA.target == sn].source)
    else:
      dsn1 = set(dA.loc[dA.target == sn].source)
      dsn2 = set(dA.loc[dA.source == sn].target)
      dsn = dsn1.intersection(dsn2)

    Dsn = np.zeros((len(NSC)))

    for ii, nsc in enumerate(NSC):
      neighbor_nodes = list(dsn.intersection(nsc[0]))

      if len(neighbor_nodes) > 0:
        Dsn[ii] = np.mean(Diss[sn, neighbor_nodes])
      else:
        Dsn[ii] = th

    # print(Dsn)

    non_trivial_covers = Dsn < th

    # print(Dsn[non_trivial_covers])

    if np.sum(non_trivial_covers) > 0:

      nn = Dsn[non_trivial_covers].shape[0]
      indx_min = np.argmin(Dsn[non_trivial_covers])
      dsn_min = Dsn[non_trivial_covers][indx_min]

      if nn > 1:
        DD = np.zeros((nn, nn))
        for kk in np.arange(nn):
          for ki in np.arange(kk+1, nn):
            DD[kk, ki] = np.abs(Dsn[non_trivial_covers][kk] - Dsn[non_trivial_covers][ki])
            DD[ki, kk] = DD[kk, ki]

        DD = linkage(squareform(DD), method="complete")
        ld = ladder_differences(DD[:, 2].ravel())
        if ld.shape[0] > 1:
          h = np.argmax(ld)
        else: h = 0
        li = cut_tree(DD, height=DD[h, 2]).ravel()
        min_point_region = li[indx_min]

      else:
        li = [0]
        min_point_region =  0

      # x=pd.DataFrame({"x" : Dsn[non_trivial_covers, 0], "y": Dsn[non_trivial_covers, 1], "label": li})
      # x["label"] = pd.Categorical(x["label"], np.unique(li))
      # sns.scatterplot(data=x, x="x", y="y", hue="label",  s=100, alpha=0.5)
      # plt.title(labels[sn])
      # plt.show()

      ii = 0
      
      for nsc, non in zip(NSC, non_trivial_covers):
        if non:
          if li[ii] == min_point_region or Dsn[non_trivial_covers][ii] == dsn_min:
            if labels[sn] not in nocs.keys():
              nocs[labels[sn]] = [nsc[1]]
              nocs_size[labels[sn]] = {nsc[1] : th - Dsn[non_trivial_covers][ii]}

            else:
              nocs[labels[sn]].append(nsc[1])
              nocs_size[labels[sn]].update({nsc[1] : th - Dsn[non_trivial_covers][ii]})
            overlap[sn] += 1
          ii += 1

  return  np.array(list(nocs.keys())), nocs, nocs_size

def discover_overlap_nodes_7(H, K : int, Cr : npt.ArrayLike, labels, undirected=False, direction="both", index="Hellinger2", **kwargs):  
  from scipy.cluster.hierarchy import cut_tree
  cr = Cr.copy()

  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)

  if direction == "source":
    overlap, nocs, nocs_size = get_nocs_information(H, Cr, dA, labels, "source", index)
  elif direction == "target":
    overlap, nocs, nocs_size = get_nocs_information(H, Cr,dA, labels, "target", index)
  elif direction == "both":
    overlap_src, nocs_src, nocs_size_src = get_nocs_information(H, Cr, dA, labels, "source", index)
    overlap_tgt, nocs_tgt, nocs_size_tgt = get_nocs_information(H, Cr, dA, labels, "target", index)

    overlap = np.hstack([overlap_src, overlap_tgt])
    overlap = np.unique(overlap)

    nocs = nocs_src.copy()

    for key, value in nocs_tgt.items():
      if key not in nocs.keys():
        nocs[key] = value
      else:
        nocs[key] += value
        nocs[key] = list(set(nocs[key]))

    nocs_size = nocs_size_src.copy()

    for key, value in nocs_size_tgt.items():
      if key not in nocs_size.keys():
        nocs_size[key] = value
      else:
        for key2, value2 in nocs_size_tgt[key].items():
          if key2 not in nocs_size[key].keys():
            nocs_size[key].update({key2 : value2})
          else:
            nocs_size[key][key2] = 0.5 * (value2 + nocs_size[key][key2])

  not_nocs = []

  for key in nocs.keys():
    if len(nocs[key]) == 1:
      not_nocs.append(key)
    i = match([key], labels)
    if len(nocs[key]) == 1 and cr[i] == -1:
      cr[i] = nocs[key][0]

  for key in not_nocs:
    del nocs[key]
    del nocs_size[key]

  return np.array(list(nocs.keys())), nocs, nocs_size, cr

def discover_overlap_nodes_8(H, K : int, Cr, labels, undirected=False, direction="source", index="Hellinger2", **kwargs):  
  from scipy.cluster.hierarchy import cut_tree, linkage
  from scipy.spatial.distance import squareform
  
  cr = Cr.copy()
  nocs_size = {}
  nocs = {}

  if direction == "source":
    if index == "Hellinger2":
      D = 1 - H.source_sim_matrix
    elif index == "D1_2_4":
      D = (1 / H.source_sim_matrix ) - 1
    else: raise ValueError("Index not implemented")
  elif direction == "target":
    if index == "Hellinger2":
      D = 1 - H.target_sim_matrix
    elif index == "D1_2_4":
      D = (1 / H.target_sim_matrix) - 1
    else: raise ValueError("Index not implemented")
  elif direction == "both":
    if index == "Hellinger2":
      a = 1 - H.source_sim_matrix
      b = 1 - H.target_sim_matrix
    elif index == "D1_2_4":
      a = (1 / H.source_sim_matrix) - 1
      b = (1 / H.target_sim_matrix) - 1
    else: raise ValueError("Index not implemented")
    keep = a != 0
    x, y = np.where(keep)
    D = np.zeros(a.shape)
    D[x, y] = np.minimum(a[x, y], b[x, y])
  else:
    raise ValueError("No accepted direction")

   ## Single nodes ----
  single_nodes = np.where(Cr == -1)[0]
  ## Nodes with single community membership ----
  NSC = [(np.where(cr == i)[0], i) for i in np.unique(cr) if i != -1]

  for sn in single_nodes:
    Dsn = np.zeros((len(NSC)+1))

    for ii, nsc in enumerate(NSC):
      Dsn[ii] = np.min(D[sn, nsc[0]])

    Dsn[-1] = 1
    non_trivial_covers = Dsn < 1

    if np.sum(non_trivial_covers) > 0:

      nn = Dsn[non_trivial_covers].shape[0]
      indx_min = np.argmin(Dsn[non_trivial_covers])
      dsn_min = Dsn[non_trivial_covers][indx_min]

      if nn > 1:
        DD = np.zeros((nn, nn))
        for kk in np.arange(nn):
          for ki in np.arange(kk+1, nn):
            DD[kk, ki] = np.abs((Dsn[non_trivial_covers][kk] - Dsn[non_trivial_covers][ki]))
            DD[ki, kk] = DD[kk, ki]

        DD = linkage(squareform(DD), method="complete")
        h = np.argmax(DD[:, 2])
        li = cut_tree(DD, height=DD[h-1, 2]).ravel()
        min_point_region = li[indx_min]

      else:
        li = [0]
        min_point_region =  0

      # x=pd.DataFrame({"x" : Dsn[non_trivial_covers], "y": [0] * Dsn[non_trivial_covers].shape[0], "label": li})
      # x["label"] = pd.Categorical(x["label"], np.unique(li))
      # sns.scatterplot(data=x, x="x", y="y", hue="label",  s=100)
      # plt.title(labels[sn])
      # plt.show()

      ii = 0
      for nsc, non in zip(NSC, non_trivial_covers):
        if non:
          if li[ii] == min_point_region or Dsn[non_trivial_covers][ii] == dsn_min:
            if labels[sn] not in nocs.keys():
              nocs[labels[sn]] = [nsc[1]]
              nocs_size[labels[sn]] = {nsc[1] : 1 - Dsn[non_trivial_covers][ii]}
            else:
              nocs[labels[sn]].append(nsc[1])
              nocs_size[labels[sn]].update({nsc[1] : 1 - Dsn[non_trivial_covers][ii]})
          ii += 1

  not_nocs = []

  for key in nocs.keys():
    if len(nocs[key]) == 1:
      not_nocs.append(key)
    i = match([key], labels)
    if len(nocs[key]) == 1 and cr[i] == -1:
      cr[i] = nocs[key][0]

  for key in not_nocs:
    del nocs[key]
    del nocs_size[key]

  return  np.array(list(nocs.keys())), nocs, nocs_size, cr

def discover_overlap_nodes_9(H, K : int, Cr, labels, direction="source",  index="Hellinger2", **kwargs):  
  from scipy.cluster.hierarchy import cut_tree, linkage
  from scipy.spatial.distance import squareform
  
  cr = Cr.copy()
  nocs_size = {}
  nocs = {}

  if direction == "source":
    if index == "D1_2_4":
      D = (1 / H.source_sim_matrix ) - 1
    else: D = 1 - H.source_sim_matrix
  elif direction == "target":
    if index == "D1_2_4":
      D = (1 / H.target_sim_matrix) - 1
    else: D = 1 - H.target_sim_matrix
  elif direction == "both":
    if index == "D1_2_4":
      a = (1 / H.source_sim_matrix) - 1
      b = (1 / H.target_sim_matrix) - 1
    else:
      a = (1 - H.source_sim_matrix)
      b = (1 - H.target_sim_matrix)
    keep = a != 0
    x, y = np.where(keep)
    D = np.zeros(a.shape)
    D[x, y] = np.minimum(a[x, y], b[x, y])
  else:
    raise ValueError("No accepted direction")

  ## Single nodes ----
  single_nodes = np.where(Cr == -1)[0]
  ## Nodes with single community membership ----
  NSC = [(np.where(cr == i)[0], i) for i in np.unique(cr) if i != -1]

  for sn in single_nodes:
    Dsn = np.zeros((len(NSC)+1))

    for ii, nsc in enumerate(NSC):
      Dsn[ii] = np.mean(D[sn, nsc[0]])

    Dsn[-1] = 1
    non_trivial_covers = Dsn < 1

    if np.sum(non_trivial_covers) > 0:

      nn = Dsn[non_trivial_covers].shape[0]
      indx_min = np.argmin(Dsn[non_trivial_covers])
      dsn_min = Dsn[non_trivial_covers][indx_min]

      if nn > 1:
        DD = np.zeros((nn, nn))
        for kk in np.arange(nn):
          for ki in np.arange(kk+1, nn):
            DD[kk, ki] = np.abs((Dsn[non_trivial_covers][kk] - Dsn[non_trivial_covers][ki]))
            DD[ki, kk] = DD[kk, ki]

        DD = linkage(squareform(DD), method="complete")
        h = np.argmax(DD[:, 2])
        li = cut_tree(DD, height=DD[h-1, 2]).ravel()
        min_point_region = li[indx_min]

      else:
        li = [0]
        min_point_region =  0

      # x=pd.DataFrame({"x" : Dsn[non_trivial_covers], "y": [0] * Dsn[non_trivial_covers].shape[0], "label": li})
      # x["label"] = pd.Categorical(x["label"], np.unique(li))
      # sns.scatterplot(data=x, x="x", y="y", hue="label",  s=100)
      # plt.title(labels[sn])
      # plt.show()

      ii = 0
      for nsc, non in zip(NSC, non_trivial_covers):
        if non:
          if li[ii] == min_point_region or Dsn[non_trivial_covers][ii] == dsn_min:
            if labels[sn] not in nocs.keys():
              nocs[labels[sn]] = [nsc[1]]
              nocs_size[labels[sn]] = {nsc[1] : 1 - Dsn[non_trivial_covers][ii]}
            else:
              nocs[labels[sn]].append(nsc[1])
              nocs_size[labels[sn]].update({nsc[1] : 1 - Dsn[non_trivial_covers][ii]})
          ii += 1

  not_nocs = []

  for key in nocs.keys():
    if len(nocs[key]) == 1:
      not_nocs.append(key)
    i = match([key], labels)
    if len(nocs[key]) == 1 and cr[i] == -1:
      cr[i] = nocs[key][0]

  for key in not_nocs:
    del nocs[key]
    del nocs_size[key]

  return  np.array(list(nocs.keys())), nocs, nocs_size, cr

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
    
def discovery_2(H, K : int, Cr, *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_2(H, K, Cr, labels, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_3(H, K : int, Cr, undirected=False, *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_3(H, K, Cr, labels, undirected=undirected, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_4(H, K : int, Cr, undirected=False, *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, sizes = discover_overlap_nodes_4(H, K, Cr, labels, undirected=undirected, **kwargs)
  return np.array([labels[i] for i in overlap]), noc_covers, sizes
  
def discovery_5(H, K : int, Cr, undirected=False, *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers = discover_overlap_nodes_5(H, K, Cr, labels, undirected=undirected, **kwargs)
  if len(overlap) > 0:
    return np.array([labels[i] for i in overlap]), noc_covers
  else:
    return np.array([]), noc_covers
  
def discovery_6(H, K : int, Cr, undirected=False, *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes = discover_overlap_nodes_6(H, K, Cr, labels, undirected=undirected, **kwargs)
  return np.array([labels[i] for i in overlap]), noc_covers, noc_sizes

def discovery_7(H, K : int, Cr : npt.ArrayLike, undirected=False, direction="both", index="Hellinger2", **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes, cr = discover_overlap_nodes_7(H, K, Cr, labels, undirected=undirected, direction=direction, index=index, **kwargs)
  return overlap, noc_covers, noc_sizes, cr

def discovery_8(H, K : int, Cr, undirected=False, direction="source", *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes, new_partition = discover_overlap_nodes_8(H, K, Cr, labels, undirected=undirected, direction=direction, **kwargs)
  return overlap, noc_covers, noc_sizes, new_partition

def discovery_9(H, K : int, Cr, undirected=False, direction="source", *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes, new_partition = discover_overlap_nodes_9(H, K, Cr, labels, undirected=undirected, direction=direction, **kwargs)
  return overlap, noc_covers, noc_sizes, new_partition