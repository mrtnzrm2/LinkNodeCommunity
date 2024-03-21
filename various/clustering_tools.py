import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from various.network_tools import *

def discover_overlap_nodes(A, Dsource, Dtarget, Cr, labels, direction="both"):  
  cr = Cr.copy()
  dA = adj2df(A)
  dA = dA.loc[dA["weight"] > 0]

  def get_nocs_information(D, Cr, dA, labels, direction):

    nocs_size = {}
    nocs = {}
    overlap = np.zeros(len(Cr))
    th = np.nanmax(D) * 1.05
    
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
          Dsn[ii] = np.mean(D[sn, neighbor_nodes])
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
              DD[kk, ki] = np.abs(Dsn[non_trivial_covers][kk] - Dsn[non_trivial_covers][ki])
              DD[ki, kk] = DD[kk, ki]

          DD = linkage(squareform(DD), method="complete")
          h = np.argmax(DD[:, 2])
          li = cut_tree(DD, height=DD[h-1, 2]).ravel()
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
                nocs_size[labels[sn]] = {nsc[1] : th - Dsn[non_trivial_covers][ii]}

              else:
                nocs[labels[sn]].append(nsc[1])
                nocs_size[labels[sn]].update({nsc[1] : th - Dsn[non_trivial_covers][ii]})
              overlap[sn] += 1
            ii += 1

    return  np.array(list(nocs.keys())), nocs, nocs_size

  if direction == "source":
    overlap, nocs, nocs_size = get_nocs_information(Dsource, Cr, dA, labels, "source")
  elif direction == "target":
    overlap, nocs, nocs_size = get_nocs_information(Dtarget, Cr, dA, labels, "target")
  elif direction == "both":
    overlap_src, nocs_src, nocs_size_src = get_nocs_information(Dsource, Cr, dA, labels, "source")
    overlap_tgt, nocs_tgt, nocs_size_tgt = get_nocs_information(Dtarget, Cr, dA, labels, "target")

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