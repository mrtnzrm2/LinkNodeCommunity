import numpy as np
import seaborn as sns
import numpy.typing as npt
from various.network_tools import *

def get_nocs_information(H, Cr, dA, labels, direction, index):
  
  from scipy.cluster.hierarchy import linkage, cut_tree
  from scipy.spatial.distance import squareform

  nocs_size = {}
  nocs = {}
  overlap = np.zeros(H.nodes)

  if index == "D1_2_4":
    if direction == "source":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.source_sim_matrix) - 1
      Diss[Diss == np.inf] = np.max(Diss[Diss < np.inf]) * 1.05
    elif direction == "target":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.target_sim_matrix) - 1
      Diss[Diss == np.inf] = np.max(Diss[Diss < np.inf]) * 1.05
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "Hellinger2":
    if direction == "source":
      Diss = np.sqrt(1 - H.source_sim_matrix)
    elif direction == "target":
      Diss = np.sqrt(1 - H.target_sim_matrix)
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "cos" or index == "corr" or index == "dot" or index == "jacw2" or index == "jacp_2":
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

def get_nocs_information_peri(H, Cr, dA : pd.DataFrame, labels, direction, index):

  '''
  Parameters
  ----------

  H : linkage matrix.
  Cr : array-like, shape (n_nodes,) is the community memebership of each node.
  '''
  
  from scipy.cluster.hierarchy import linkage, cut_tree
  from scipy.spatial.distance import squareform

  nocs_size = {}
  nocs = {}

  if index == "D1_2_4" or index == "dist_sim":
    if direction == "source":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.source_sim_matrix) - 1
      Diss[Diss == np.inf] = np.max(Diss[Diss < np.inf]) * 1.05
    elif direction == "target":
      np.seterr(divide='ignore', invalid='ignore')
      Diss = (1 / H.target_sim_matrix) - 1
      Diss[Diss == np.inf] = np.max(Diss[Diss < np.inf]) * 1.05
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "Hellinger2":
    if direction == "source":
      Diss = np.sqrt(1 - H.source_sim_matrix)
    elif direction == "target":
      Diss = np.sqrt(1 - H.target_sim_matrix)
    else:
      raise ValueError("No accepted direction in discovery channel")
  elif index == "cos" or index == "corr" or index == "dot" or index == "jacw2":
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

  th = np.max(Diss) + 1e-3
  
  ## Single nodes ----
  single_nodes = np.where(Cr == -1)[0]

  ## Nodes with single community membership ----
  NSC = [(set(np.where(Cr == i)[0]), i) for i in np.sort(np.unique(Cr)) if i != -1]

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

    non_trivial_covers = Dsn < th

    if np.sum(non_trivial_covers) > 0:

      nn = Dsn[non_trivial_covers].shape[0]
      indx_min = np.argmin(Dsn[non_trivial_covers])
      dsn_min = Dsn[non_trivial_covers][indx_min]

      if nn > 1:
        DD = np.zeros((nn, nn))
        for kk in np.arange(nn):
          for ki in np.arange(kk+1, nn):

            DD[kk, ki] = Dsn[non_trivial_covers][kk] + Dsn[non_trivial_covers][ki] + \
              np.mean(Diss[np.array(list(NSC[kk][0])), :][:, np.array(list(NSC[ki][0]))])
            
            DD[ki, kk] = DD[kk, ki]

        # if labels[sn] == "orbvl":
        #   _ = plt.hist(squareform(DD))

        Dh = linkage(squareform(DD), method="complete")
          
        ld = ladder_differences(Dh[:, 2].ravel())
        if ld.shape[0] > 1:
          h = np.argmax(ld)
        else: h = 0
        li = cut_tree(Dh, height=Dh[h, 2]).ravel()

        # sh4 = compute_Sh4(Dh, Dh.shape[0]+1, use_tqdm=False)
        # nk4, _ = drawly_Sh(sh4, on=False)
        # li = cut_tree(Dh, n_clusters=nk4).ravel()
        
        # if labels[sn] == "aip" and direction == "target":
        #   print(indx_min)
        #   _, ax = plt.subplots(2)
        #   _ = dendrogram(Dh, color_threshold=Dh[len(NSC) - nk4 - 1, 2], ax=ax[0])
        #   _ = ax[1].hist(squareform(DD))
        #   ax[1].axvline(Dh[len(NSC) - nk4 - 1, 2], color='r', linestyle='--', label="Cut-off")

        # if labels[sn] == "aip" and direction == "source":
        #   print(indx_min)
        #   _, ax = plt.subplots(2)
        #   _ = dendrogram(Dh, color_threshold=Dh[len(NSC) - nk4 - 1, 2], ax=ax[0])
        #   _ = ax[1].hist(squareform(DD))
        #   ax[1].axvline(Dh[len(NSC) - nk4 - 1, 2], color='r', linestyle='--', label="Cut-off")

        
        min_point_region = li[indx_min]

      else:
        li = [0]
        min_point_region =  0

      ii = 0
      
      for nsc, non in zip(NSC, non_trivial_covers):
        if non:
          if li[ii] == min_point_region or Dsn[non_trivial_covers][ii] == dsn_min:
            if labels[sn] not in nocs.keys():
              nocs[labels[sn]] = [nsc[1]]
              nocs_size[labels[sn]] = {nsc[1] : 1 - Dsn[non_trivial_covers][ii]**2}

            else:
              nocs[labels[sn]].append(nsc[1])
              nocs_size[labels[sn]].update({nsc[1] : 1 - Dsn[non_trivial_covers][ii]**2})
          ii += 1

  return  np.array(list(nocs.keys())), nocs, nocs_size

def discover_overlap_nodes_8(H, K : int, Cr : npt.ArrayLike, labels, undirected=False, direction="both", index="Hellinger2", **kwargs):  
  from scipy.cluster.hierarchy import cut_tree
  import copy

  cr = Cr.copy()

  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)

  if direction == "source":
    _, nocs, nocs_size = get_nocs_information_peri(H, Cr, dA, labels, "source", index)
  elif direction == "target":
    _, nocs, nocs_size = get_nocs_information_peri(H, Cr,dA, labels, "target", index)
  elif direction == "both":
    _, nocs_src, nocs_size_src = get_nocs_information_peri(H, Cr, dA, labels, "source", index)
    _, nocs_tgt, nocs_size_tgt = get_nocs_information_peri(H, Cr, dA, labels, "target", index)

    nocs = copy.deepcopy(nocs_src)  # Start with source NOCs

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

  no_nocs = []

  for key in nocs.keys():
    if len(nocs[key]) == 1:
      no_nocs.append(key)
    i = match([key], labels)
    if len(nocs[key]) == 1 and cr[i] == -1:
      cr[i] = nocs[key][0]

  for key in no_nocs:
    del nocs[key]
    del nocs_size[key]


  # Get directions
  nocs_direction = {}
  ucr = np.unique(cr)
  lucr = len(ucr)

  for key in nocs.keys():
      
    noc_mem = np.array(nocs[key])     # NOC memberships
    
    if -1 in ucr:
      dir_key = np.zeros(lucr+1, dtype=int)
    else:
      dir_key = np.zeros(lucr, dtype=int)

    if direction == "source":
      if -1 in ucr:
        dir_key[noc_mem + 1] = 1   # Source direction
      else:
        dir_key[noc_mem] = 1

    elif direction == "target":
      if -1 in ucr:
        dir_key[noc_mem + 1] = 2  # Target direction
      else:
        dir_key[noc_mem] = 2

    else:
      source_key = np.sort(np.array(nocs_src[key]))
      target_key = np.sort(np.array(nocs_tgt[key]))
      both_key = np.intersect1d(source_key, target_key, assume_unique=True)

      if -1 in ucr:
        dir_key[source_key + 1] = 1  # Source direction
        dir_key[target_key + 1] = 2  # Target direction
        dir_key[both_key + 1] = 3  # Both directions

      else:

        dir_key[source_key] = 1
        dir_key[target_key] = 2
        dir_key[both_key] = 3

    nocs_direction[key] = dir_key

    
  return np.array(list(nocs.keys())), nocs, nocs_size, nocs_direction, cr


def discovery_7(H, K : int, Cr : npt.ArrayLike, undirected=False, direction="both", index="Hellinger2", **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes, cr = discover_overlap_nodes_7(H, K, Cr, labels, undirected=undirected, direction=direction, index=index, **kwargs)
  return overlap, noc_covers, noc_sizes, cr

def discovery_8(H, K : int, Cr, undirected=False, direction="source", *args, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, nocs_cover, nocs_sizes, nocs_direction, new_partition  = discover_overlap_nodes_8(H, K, Cr, labels, undirected=undirected, direction=direction, **kwargs)
  return overlap, nocs_cover, nocs_sizes, nocs_direction, new_partition

def renormalization(rlabels : npt.ArrayLike, R : npt.NDArray, D : npt.NDArray, labels : npt.ArrayLike):
  '''
  Parameters
  ----------
  
  rlabels : array-like, shape (n_nodes,) is the community memebership of each node.
  R : array-like, shape (n_nodes, n_nodes) is the (weighted) adjacency matrix.
  D : array-like, shape (n_nodes, n_nodes) is the distance matrix.
  labels : array-like, shape (n_nodes,) is the label of each node.
  '''

  from networks.toy import TOY
  from modules.hierarmerge import Hierarchy
  from modules.colregion import colregion
  from collections import Counter

  # Renormalize the R matrix using the rlabels

  skimmed_rlabels = skim_partition(rlabels)
  map_skimmed_rlabels = {r : sk for r, sk in zip(rlabels, skimmed_rlabels)}

  single_communities = Counter(rlabels)
  single_communities_rlabels = np.array([k for k, v in single_communities.items() if v == 1])
  single_communities = np.array([labels[rlabels == k][0] for k, v in single_communities.items() if v == 1])

  unique_rlabels = np.sort(np.unique(rlabels))

  Rnorm = np.zeros((unique_rlabels.shape[0], unique_rlabels.shape[0]))
  Dnorm = np.zeros((unique_rlabels.shape[0], unique_rlabels.shape[0]))

  for i, k in enumerate(unique_rlabels):
    for j, l in enumerate(unique_rlabels):
      if i != j:
        Rnorm[i, j] = np.mean(R[rlabels == k, :][:, rlabels == l])
        Dnorm[i, j] = np.mean(D[rlabels == k, :][:, rlabels == l])
      else:
        nkk = np.sum(rlabels == k)
        if nkk > 1:
          Rnorm[i, j] = np.sum(R[rlabels == k, :][:, rlabels == k]) / (nkk*(nkk - 1))
          Dnorm[i, j] = np.sum(D[rlabels == k, :][:, rlabels == k]) / (nkk*(nkk - 1))

  # tmp = Rnorm.copy()
  # tmp[tmp == 0] = np.nan
  # tmp[tmp > 0] = np.log10(tmp[tmp > 0])
  # sns.heatmap(tmp)

  NET = TOY(Rnorm, "single", labels=unique_rlabels, index="Hellinger2", discovery="discovery_9")
  H = Hierarchy(NET, Rnorm, Rnorm, Dnorm, unique_rlabels.shape[0], "single", "ZERO", chardist=False)

  ## Compute features ----
  H.BH_features_cpp_no_mu()

  ## Compute la arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])

  ## Compute the dendrogram ----
  L = colregion(NET, labels=NET.labels)
  L.get_regions()
  H.set_colregion(L)

  k, r, _ = get_best_kr_equivalence("_S", H)
  post_rlabels = get_labels_from_Z(H.Z, r[0])

  # RN = Rnorm.copy()
  # RN[RN == 0] = np.nan
  # RN[RN > 0] = np.log10(RN[RN > 0])

  # heatmap_dendro(r[0], H.Z, RN, labels=unique_rlabels)
  # lcmap_dendro(k[0], H.H, r[0], H.Z, H.dA, labels=unique_rlabels, on=True)
  # plot_measurements_S(H.BH, on=True)

  nocs = {lab : None for lab in single_communities}

  for lab, s in zip(single_communities, single_communities_rlabels):
    # print(lab, s)
    renormalized_communities = post_rlabels[unique_rlabels == s][0]
    nocs[lab] = [map_skimmed_rlabels[r] for r in unique_rlabels[post_rlabels == renormalized_communities] if map_skimmed_rlabels[r] != -1]
  
  nocs_size = {lab : None for lab in single_communities}

  for lab, v in nocs.items():
    nocs_size[lab] = np.ones(len(v)) / len(v)
      
  nocs_directions = {lab : None for lab in single_communities}
  
  for lab, v in nocs.items():
    nocs_directions[lab] = np.ones(len(v)) * 3

  return list(nocs.keys()), nocs, nocs_size, nocs_directions, skimmed_rlabels
