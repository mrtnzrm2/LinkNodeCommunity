import numpy as np
import numpy.typing as npt
import pandas as pd
pd.options.mode.chained_assignment = None
from os.path import join, exists, isfile
from sklearn.metrics import adjusted_mutual_info_score
from scipy.special import factorial
from scipy.stats import hypergeom
import math
from os import remove, stat
import pickle as pk
from various.omega import Omega
import matplotlib.colors as mc
import seaborn as sns

def make_cmap_from_K(K, trivial="-1", cmap="hls", seed=None, numeric=True):
  '''
  Create a dictionary from a list of community memberships K, where each unique
  element in K is associated with a color from palette cmap. Notice, K is expected
  to have numeric strings, but it also can handle other formats. The trivial element '-1'
  is especial and it is linked to the gray color.
  '''
  from matplotlib.colors import to_hex
  if numeric:
    uK = np.sort(np.unique(K.astype(int))).astype(str)
  else:
    uK = np.sort(np.unique(K)).astype(str)

  if np.isin(trivial, uK):
      if seed is None:
        cm = list(sns.color_palette(cmap, uK.shape[0]-1))
      else:
        print(f"Seed: {seed}")
        np.random.seed(seed)
        n = len(uK)
        cm = list(np.array(sns.color_palette(cmap, uK.shape[0]-1))[np.random.permutation(np.arange(n-1))])
      cm = [to_hex((0.5, 0.5, 0.5))] + cm
  else:
    if seed is None:
      cm = list(sns.color_palette(cmap, uK.shape[0]))
    else:
      print(f"Seed: {seed}")
      np.random.seed(seed)
      n = len(uK)
      cm = list(np.array(sns.color_palette(cmap, uK.shape[0]))[np.random.permutation(np.arange(n))])
    
  cm = {u: to_hex(c) for u, c in zip(uK, cm)}
  return cm

def custom_cut_tree(H : npt.NDArray, n_clusters=None, height=None):
  '''
  Similar to scipy.cluster.hierarchy function cut_tree, but more optimized.

  '''

  if n_clusters is None and height is None:
    raise ValueError("n_clusters or height must be given.")
  elif n_clusters is not None and height is None:
    thrd = n_clusters
    thrd_t = 0
  elif height is not None and n_clusters is None:
    thrd = height
    thrd_t = 1
  else:
     pass
    
  N = H.shape[0]+1
  T = {(i) : [i] for i in np.arange(N)}

  K = N
  i = 0
  while True:
    if thrd_t == 0:
      if K <= thrd: break
    else:
      if H[i, 2] > thrd: break

    nx, ny = int(H[i, 0]), int(H[i, 1])

    T[(N+i)] = T[(nx)] + T[(ny)]
    
    del T[(nx)]
    del T[(ny)]
    
    i += 1
    K -= 1
  
  partition = np.zeros(N, dtype=np.int64)
  for key, val in T.items():
    members = np.array(val)
    partition[members] = key
  
  return partition

# D_min, D_max = np.min(D), np.max(D)
# dD = (D_max - D_min) / nbin
# d_range = np.arange(D_min, D_max + 1e-5, dD / 2)
# d_range[-1] += 1e-5

# src_bin = [np.nanmean(src[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# src_bin += [np.nanmean(src[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# src_bin += [np.nanmean(src[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# src_bin = np.array(src_bin)

# tgt_bin = [np.nanmean(tgt[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# tgt_bin += [np.nanmean(tgt[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# tgt_bin += [np.nanmean(tgt[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# tgt_bin = np.array(tgt_bin)

# src_inset_bin = [np.nanmean(src_inset[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# src_inset_bin += [np.nanmean(src_inset[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# src_inset_bin += [np.nanmean(src_inset[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# src_inset_bin = np.array(src_inset_bin)

# tgt_inset_bin = [np.nanmean(tgt_inset[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# tgt_inset_bin += [np.nanmean(tgt_inset[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# tgt_inset_bin += [np.nanmean(tgt_inset[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# tgt_inset_bin = np.array(tgt_inset_bin)

# var_src_inset_bin = [np.nanvar(src_inset[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# var_src_inset_bin += [np.nanvar(src_inset[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# var_src_inset_bin += [np.nanvar(src_inset[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# var_src_inset_bin = np.array(var_src_inset_bin)

# var_tgt_inset_bin = [np.nanvar(tgt_inset[np.where((D >= d_range[0]) & (D < d_range[1]))[0]])]
# var_tgt_inset_bin += [np.nanvar(tgt_inset[np.where((D >= d_range[i]) & (D < d_range[i+2]))[0]]) for i in np.arange(1, 2 * nbin + 1e-5)[::2][:-1].astype(int)]
# var_tgt_inset_bin += [np.nanvar(tgt_inset[np.where((D >= d_range[-2]) & (D < d_range[-1]))[0]])]
# var_tgt_inset_bin = np.array(var_tgt_inset_bin)

def ladder_differences(d : npt.ArrayLike) -> npt.ArrayLike:
  return np.array([d[i+1] - d[i] for i in np.arange(d.shape[0]-1)])

def formating_Z2HMI(Z : npt.NDArray, nodes : int):
  relabels = np.arange(nodes, 2*(nodes-1)+1)

  rZ = np.zeros((Z.shape[0], Z.shape[1]+3))
  rZ[:, :-3] = Z
  rZ[:, -3] = relabels

  c = 0
  for i in np.arange(1, nodes-1):
      if rZ[i, 2] > rZ[i-1, 2]:
        c += 1
      rZ[i, -2] = c

  for i in np.arange(1, nodes-1):
      if  rZ[nodes-1-(i+1), -2] == rZ[nodes-1-i, -2]:
          rZ[nodes-1-(i+1), -1] = rZ[nodes-1-i, -1] + 1

  tree = []

  def insert(tree, k1, k2, r, nodes):
      if isinstance(tree, list):
          if len(tree) == 0 or r in tree:
              if k1 >= nodes or k2 >= nodes:
                tree.append([k1])
                tree.append([k2])
              elif k1 < nodes and k2 < nodes:
                tree.append(k1)
                tree.append(k2)
              else:
                tree.append([k1, k2])
          else:
              for t in tree:
                insert(t, k1, k2, r, nodes)

  def insert2(tree, k1, k2, r, nodes):
      if isinstance(tree, list):
          if len(tree) == 0 or r in tree:
              if k1 >= nodes and k2 >= nodes:
                tree.append([k1])
                tree.append([k2])
              elif k1 < nodes and k2 < nodes:
                tree.append(k1)
                tree.append(k2)
              elif k1 < nodes and k2 >= nodes:
                tree.append(k1)
                tree.append([k2])
              else:
                tree.append([k1])
                tree.append(k2)
          else:
              for t in tree:
                insert2(t, k1, k2, r, nodes)
  for i in np.arange(nodes-1):
      k1 = int(rZ[nodes-2-i, 0])
      k2 = int(rZ[nodes-2-i, 1])
      r = int(rZ[nodes-2-i, 4])

      if rZ[nodes-2-i, -1] > 0 and i > 0:
        J = np.where(rZ[:, -2] == rZ[nodes-2-i, -2])[0]
        f = False
        for j in J:
            if j > nodes-2-i and r == rZ[j, 1] and rZ[j, 0] < nodes:
              r = int(rZ[j, 0])
              insert2(tree, k1, k2, r, nodes)
              f = True
              break
        if not f:
          insert(tree, k1, k2, r, nodes)
      else:
        insert(tree, k1, k2, r, nodes)
  
  def remove(tree, r):
      if isinstance(tree, list):
        if r in tree:
            tree.remove(r)
        else:
            for t in tree:
              remove(t, r)
  
  def count_empty(tree, n):
     if isinstance(tree, list):
        if [] in tree:
          n += 1
        else:
          for t in tree:
              count_empty(t, n)

  def remove_empty(tree):
      if isinstance(tree, list):
        if [] in tree:
            tree.remove([])
            return
        else:
            for t in tree:
              remove_empty(t)

  for z in -np.sort(-rZ[:, 4])[1:]:
      remove(tree, int(z))

  n = np.array([0])
  count_empty(tree, n)
  while n[0] != 0:
      remove_empty(tree)
      n = np.array([0])
      count_empty(tree, n)

  def check_mix(tree):
    fi = 0
    fl = 0
    for t in tree:
      if isinstance(t, int): fi += 1
      if isinstance(t, list): fl += 1
      if fi > 0 and fl > 0: return True
    return False
  
  def check_list(tree):
    for t in tree:
      if not isinstance(t, list): return False
    return True
  
  def repack(tree, id):
    if isinstance(tree, list):
      if len(tree) > 1 and check_mix(tree):
        for i, t in enumerate(tree):
          if t == id:
            tree[i] = [id]
          else:
            repack(t, id)
      elif check_list(tree):
        for t in tree:
          repack(t, id)

  # from various.hit import check
  # print(check(tree))
  # print(tree, "\n", "\n")


  for i in range(nodes):
    repack(tree, i)

  return tree

def draw_brace(ax, xspan, yy, text, color="black"):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color=color, lw=1)

    ax.text((xmax+xmin)/2., yy+.04*yspan, text, ha='center', va='bottom', color=color)

def adj2Den(A, dir=False):
    n = A.shape[0]
    if dir:
      m = np.sum(A > 0) / 2
      return m / (n * (n - 1) / 2)
    else:
      m = np.sum(A > 0)
      return m / (n * (n - 1))
    
def adj2barW(A, dir=False):
    w = A[A!=0]
    return np.mean(w)


def random_partition_R(m : int, R : int, labels : npt.ArrayLike) -> dict:
  A_nodes = np.arange(m)
  if R > m:
    raise RuntimeError("Number of communities more than the number of nodes.")
  
  initial_nodes = np.random.choice(A_nodes, size=R, replace=False)
  partition = {i:[labels[n]] for i, n in enumerate(initial_nodes)}

  # Take out the initial nodes assigned to each urn.
  A_nodes = np.array([n for n in A_nodes if n not in initial_nodes])
  A_nodes_memberships = np.random.choice(np.arange(R), size=A_nodes.shape[0], replace=True)

  for mem, n in zip(A_nodes_memberships, A_nodes):
    partition[mem].append(labels[n])
  
  return partition

def pvalue2asterisks(pvalue):
  if  not np.isnan(pvalue): 
    if pvalue > 0.05:
      a = "n.s."
    elif pvalue <= 0.05 and pvalue > 0.001:
      a = "*" 
    elif pvalue <= 0.001 and pvalue > 0.0001:
      a = "**" 
    else:
      a = "***"
  else:
    a = "nan"
  return a

def hungarian_algorithm(cost_matrix : npt.NDArray):
    num_rows, num_cols = cost_matrix.shape
    num_workers = num_rows
    num_jobs = num_cols

    if num_workers > num_jobs:
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, num_workers - num_jobs)), mode='constant')
    elif num_jobs > num_workers:
        cost_matrix = np.pad(cost_matrix, ((0, num_jobs - num_workers), (0, 0)), mode='constant')

    num_rows, num_cols = cost_matrix.shape

    marked_zeros = np.zeros((num_rows, num_cols))
    row_covered = np.zeros(num_rows, dtype=bool)
    col_covered = np.zeros(num_cols, dtype=bool)
    num_covered = 0

    while num_covered < num_workers:
        marked_zeros.fill(0)
        for i in range(num_rows):
            for j in range(num_cols):
                if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                    marked_zeros[i, j] = 1

        if np.sum(marked_zeros) == 0:
            min_val = np.min(cost_matrix[~row_covered, :][:, ~col_covered])
            cost_matrix[~row_covered, :][:, ~col_covered] -= min_val
            row_min = np.min(cost_matrix[~row_covered, :], axis=1)
            cost_matrix[~row_covered, :] -= row_min[:, np.newaxis]
            col_min = np.min(cost_matrix[:, ~col_covered], axis=0)
            cost_matrix[:, ~col_covered] -= col_min
        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    if marked_zeros[i, j]:
                        if not row_covered[i] and not col_covered[j]:
                            marked_zeros[i, j] = 2
                            row_covered[i] = True
                            col_covered[j] = True
                            num_covered += 1
                            break

    assignment = []
    for i in range(num_workers):
        for j in range(num_jobs):
            if marked_zeros[i, j] == 2:
                assignment.append((i, j))

    return assignment

def cover_alignment(cov : dict, cov_ref : dict):
  cost = np.zeros((len(cov), len(cov)))
  for i, (k, ele) in enumerate(cov.items()):
    for j, (kf, elef) in enumerate(cov_ref.items()):
      cost[i, j] = 1 / (1 + len(set(ele).intersection(set(elef))))
  
  min_cost = np.min(cost)
  cost *= min_cost

  alignment = hungarian_algorithm(cost.astype(int))
  return {alignment[k][1]: ele for k, ele in cov.items()}

def Requiv2K(H : npt.NDArray, r):
  return np.min(H[H[:, 1] == r, 0]).astype(int)

def cover_node_2_node_cover(cover : dict, labels):
    node_cover = {k: [] for k in labels}
    for k, vals in cover.items():
      for l in labels:
        if l in vals:
          node_cover[l].append(k)
    
    return node_cover

def hex2(s):
  return mc.to_hex(s)

def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def fast_cut_tree(H : npt.NDArray, n_clusters=None, height=None):
  '''
  Similar to scipy.cluster.hierarchy function cut_tree, but optimized.
    Parameters
    ----------

    H : npt.NDArray
        Hierarchical clustering linkage matrix.
    n_clusters : int, optional
        Number of clusters to form. If None, height must be specified.
    height : float, optional
        Threshold to apply when forming clusters. If None, n_clusters must be specified.

    Returns
    -------

    partition : npt.NDArray
        Array of cluster labels for each node in the hierarchy.
  '''

  if H.ndim != 2 or H.shape[1] != 4:
    raise ValueError("H must be a linkage matrix with shape (n-1, 4).")

  if n_clusters is None and height is None:
    raise ValueError("n_clusters or height must be given.")
  elif n_clusters is not None and height is None:
    thrd = n_clusters
    thrd_t = 0
  elif height is not None and n_clusters is None:
    thrd = height
    thrd_t = 1
  else:
     pass
    
  N = H.shape[0]+1
  T = {(i) : [i] for i in np.arange(N)}

  K = N
  i = 0

  while True:
    if thrd_t == 0:
      if K <= thrd: break
    else:
      if H[i, 2] > thrd: break

    nx, ny = int(H[i, 0]), int(H[i, 1])

    T[(N+i)] = T[(nx)] + T[(ny)]
    
    del T[(nx)]
    del T[(ny)]
    
    i += 1
    K -= 1
  
  partition = np.zeros(N, dtype=np.int64)
  for key, val in T.items():
    members = np.array(val)
    partition[members] = key
  
  return partition

def linear_partition(partition : npt.ArrayLike):
  ''' 
  Renumber the communities linearly. From 0 to number of communities - 1.
  Parameters
  ----------

  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------

  npt.NDArray
      Renumbered partition with labels from 0 to number of communities - 1.'''
  
  par = partition.copy()
  new_partition = par
  ndc = np.unique(par)
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def collapsed_partition(partition : npt.ArrayLike):
  ''' Renumber the communities linearly. From 0 to number of communities - 1. Singletons are replaced by -1.

  Parameters
  ----------

  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------

  npt.NDArray
      Renumbered partition with labels from 0 to number of communities - 1. Singletons are replaced by -1.'''
  par = partition.copy()
  from collections import Counter
  fq = Counter(par)
  for i in fq.keys():
    if fq[i] == 1: par[par == i] = -1
  new_partition = par
  ndc = np.unique(par[par != -1])
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def skim_partition(partition : npt.ArrayLike):
  par = partition.copy()
  from collections import Counter
  fq = Counter(par)
  for i in fq.keys():
    if fq[i] == 1: par[par == i] = -1
  new_partition = par
  ndc = np.unique(par[par != -1])
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def Dc_id(dA, id, undirected=False):
  # Filter dataframe ----
  if not undirected:
    dAid = dA.loc[dA.id == id]
  else:
    dAid = dA.loc[(dA.id == id) & (dA.source > dA.target)]
  # Get source nodes ----
  src = set([i for i  in dAid.source])
  # Get target nodes list ----
  tgt = set([i for i in dAid.target])
  # Get number of edges ----
  m = dAid.shape[0]
  # Compute Dc ----
  n = len(tgt.union(src))
  if ~undirected:
    if n > 1 and m >= n: return (m - n + 1) / (n - 1) ** 2
    else: return 0
  else:
   if n > 1 and m >= n: return (m - n + 1) / (n * (n - 1) / 2. - n + 1.)
   else: return 0

def minus_one_Dc(dA, undirected=False):
  ids = np.sort(
    np.unique(
      dA["id"].to_numpy()
    )
  )
  for id in ids:
    Dc = Dc_id(dA, id, undirected=undirected)
    if Dc <= 0:
      dA.loc[dA["id"] == id, "id"] = -1

def range_and_probs_from_DC(D, CC, bins):
  nodes = CC.shape[1]
  # C = CC.copy()[:nodes, :nodes]
  # D_ = D[:nodes, :nodes]

  C = CC.copy()
  D_ = D[:, :nodes]

  # Treat distances ----
  min_d = np.min(D_[D_ > 0])
  max_d = np.max(D_)
  d_range = np.linspace(min_d, max_d, bins+1)

  d_range[-1] += 1e-2

  # Bin size - delta ----
  delta = (d_range[1] - d_range[0]) / 2
  counts = np.zeros(bins)

  # Sum counts ----
  for i in np.arange(C.shape[0]):
    for j in np.arange(C.shape[1]):
      if i == j: continue
      d = D_[i, j]
      for k in np.arange(bins):
        if d_range[k] <= d and d_range[k + 1] > d:
          counts[k] += C[i, j]
          break

  # Prepare x and y ----
  y = counts / (np.sum(counts) * 2 * delta)
  x = d_range + delta
  x = x[:-1]

  # print(y)
  # print(x)
  # raise ValueError("")

  return d_range, x, y

def get_best_kr(score, H, undirected=False, mapping="X_diag"):
  k = 1
  if score == "_D":
    k = get_k_from_D(
      get_H_from_BH(H)
    )
    k = [k]
  elif score == "_X":
    k = get_k_from_X(
      get_H_from_BH(H), order=0
    )
    k = [k]
  elif score == "_S":
    k = get_k_from_S(
      get_H_from_BH(H)
    )
  elif score == "_SD":
    k = get_k_from_SD(
      get_H_from_BH(H)
    )
  else: raise ValueError(f"Unexpected score: {score}")
  if not isinstance(k, list): k = [k]
  r = get_r_from[mapping](
    k, H.H, H.Z, H.A, H.nodes, undirected=undirected
  )
  if isinstance(r, list) and isinstance(k, list):
    return np.array(k), np.array(r)
  elif isinstance(k, list):
    return np.array(k), np.array([r])
  elif isinstance(r, list):
    return np.array([k]), np.array(r)
  else:
    return np.array([k]), np.array([r])

def get_best_kr_equivalence(score, H):
  k = 1
  if score == "_D":
    k, h = get_k_from_D(
      get_H_from_BH(H)
    )
  elif score == "_X":
    k, h = get_k_from_X(
      get_H_from_BH(H), order=0
    )
  elif score == "_S":
    k, h = get_k_from_S(
      get_H_from_BH(H)
    )
  elif score == "_S2":
    k, h = get_k_from_S(
      get_H_from_BH(H)
    )
  elif score == "_SD":
    k, h = get_k_from_SD(
      get_H_from_BH(H)
    )
  else: raise ValueError(f"Unexpected score: {score}")
  r = get_r_from_equivalence(k, H)
  # print(k)
  # k = int(k)
  if isinstance(r, list) and isinstance(k, list):
    return np.array(k), np.array(r), np.array(h)
  elif isinstance(k, list):
    return np.array(k), np.array([r]), np.array(h)
  elif isinstance(r, list):
    return np.array([k]), np.array(r), np.array([h])
  else:
    return np.array([k]), np.array([r]), np.array([h])

def aesthetic_ids_vector(v):
  vv = v.copy()
  ids = np.sort(np.unique(v))
  if -1 in ids:
    ids = ids[1:]
    aids = np.arange(len(ids))
  else:
    aids = np.arange(len(ids))
  for i, id in enumerate(ids):
    vv[v == id] = aids[i]
  return vv

def aesthetic_ids(dA):
    ids = np.sort(np.unique(dA["id"].to_numpy()))
    if ids.shape[0] == 1 and ids[0] == -1:
      pass
    else:
      aids = np.zeros(ids.shape[0])
      if -1 in ids:
        aids[0] = -1
        aids[1:] = np.arange(1, len(ids[1:]) + 1)
      else:
        aids = np.arange(1, len(ids)+1)
      dump = np.zeros(dA.shape[0])
      for i, id in enumerate(ids):
        dump[dA["id"] == id] = aids[i]
      dA["id"] = dump.astype(int)

def combine_dics(f1, f2):
  for k in f2.keys():
    if k not in f1.keys():
      f1[k] = f2[k]
    else:
      f1[k] += f2[k]

def bar_data(dA, nodes, labels, norm=False):
  # unique ids ----
  Tids = np.unique(dA["id"])
  # Create data ----
  data = pd.DataFrame()
  # Start loop ----
  from collections import Counter
  for i in np.arange(nodes):
    ## Source ----
    ids_out = dA.id.loc[dA["source"] == i].to_numpy().ravel()
    fq = dict(Counter(ids_out))
    ## Target ----
    ids_in = dA.id.loc[dA["target"] == i].to_numpy().ravel()
    fq_in = dict(Counter(ids_in))
    # Combine dictionaries ----
    combine_dics(fq, fq_in)
    # Keys ----
    keys = list(fq.keys())
    if len(keys) == 0: continue
    # keys = np.array(keys) - 1
    keys = [
      np.where(Tids == id)[0][0] for id in keys
    ]
    # Values ----
    values = list(fq.values())
    values = np.array(values)
    # Sizes ----
    size_ids = np.zeros(len(Tids))
    size_ids[keys] = values
    if norm:
      size_ids /= np.sum(size_ids)
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            # 'node_index' : np.array([i] * len(Tids)), 
            "nodes" : np.array([labels[i]] * len(Tids)),
            "ids" : Tids,
            "size" : size_ids
          }
        )
      ],
      ignore_index=True
    )
  return data

def reverse_partition(Cr, labels):
  s = np.sort(np.unique(Cr).astype(int))
  s = s[s != -1]
  k = {r : [] for r in s}
  for i, r in enumerate(Cr):
    if r == -1: continue
    k[r].append(labels[i])
  return k

def nocs2parition(partition: dict, nocs: dict):
  for noc in nocs.keys():
    for cover in nocs[noc]:
      if str(noc) not in partition[cover]:
        partition[cover].append(str(noc))

def get_H_from_BH(H):
  h = pd.DataFrame()
  for i in np.arange(len(H.BH)):
    h  = pd.concat(
      [h , H.BH[i]],
      ignore_index=True
    )
  return h

def get_H_from_BH_with_maxmu(H):
  h = pd.DataFrame()
  for i in np.arange(len(H.BH)):
    h = pd.concat([h, H.BH[i]], ignore_index=True)
  h = h.groupby(["K", "D", "S"])["mu"].max().reset_index()
  return h

def get_k_from_ntrees(H):
  k = H["K"].loc[
  H["ntrees"] == 0
  ]
  if (len(k) > 1):
    print("warning: more than one k")
    k = np.max(k)
  return k

def get_k_from_mu(H):
  k = H["K"].loc[
    H["mu"] == np.nanmax(H["mu"])
  ]
  if (len(k) > 1):
    print("warning: more than one k")
    k = k.iloc[0]
  return k

def get_k_from_avmu(H):
  avH = H.groupby(["K"]).mean()
  k = avH.index[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if len(k) > 0:
    k = k[0]
  return k

def get_k_from_X(H, order=0):
  target_maximum = np.sort(H["X"])
  target_maximum = target_maximum[-1 - order]
  k = H.index[
    H["X"] == target_maximum
  ].to_numpy().reshape(-1).astype(int)
  h = H.height[
    H["X"] == target_maximum
  ].to_numpy().reshape(-1).astype(int)
  if k.shape[0] > 1:
    k = k[0]
    h = h[0]
  return int(k), float(h)

def get_labels_from_Z(Z : npt.NDArray, r : int):
  save_Z = np.sum(Z, axis=1)
  if 0 in save_Z: return np.array([np.nan])
  labels = fast_cut_tree(
    Z,
    n_clusters=r
  )
  return labels

def get_r_from_mu(H):
  r = H["NAC"].loc[
    H["mu"] == np.nanmax(H["mu"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return r

def get_r_from_D(H):
  r = H["NAC"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return r

def get_k_from_D(H : pd.DataFrame):
  r = H["K"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  h = H["height"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  if r.shape[0] > 1:
    print("warning: more than one k")
    r = r.iloc[0]
    h = h.iloc[0]
  return int(r), float(h)

def   get_k_from_S(H : pd.DataFrame):
  r = H["K"].loc[
    H["S"] == np.nanmax(H["S"])
  ]
  h = H["height"].loc[
    H["S"] == np.nanmax(H["S"])
  ]
  if r.shape[0] > 1:
    print(">>> warning: more than one k")
  return int(r.iloc[-1]), float(h.iloc[-1])

def   get_k_from_S2(H : pd.DataFrame):
  r = H["K"].loc[
    H["S2"] == np.nanmax(H["S2"])
  ]
  h = H["height"].loc[
    H["S2"] == np.nanmax(H["S2"])
  ]
  if r.shape[0] > 1:
    print(">>> warning: more than one k")
  return int(r.iloc[0]), float(h.iloc[0])

def get_k_from_SD(H):
  if "SD" in H.columns:
    r = H["K"].loc[H.SD == np.nanmax(H.SD)]
    h = H["height"].loc[H.SD == np.nanmax(H.SD)]
  else:
    best = (H.D / np.nansum(H.D))* (H.S / np.nansum(H.S))
    r = H.K.loc[best == np.nanmax(best)]
    h = H.height.loc[best == np.nanmax(best)]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
    h = h.iloc[0]
  return int(r), h

def get_r_from_avmu(H):
  avH = H.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_maxmu(H):
  avH = H.groupby(["K", "alpha"]).max()
  avH = avH.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_X(H):
  avH = H.groupby(["K", "n"]).max()
  avH = avH.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["X"] == np.nanmax(avH["X"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_equivalence(k, H):
  if isinstance(k, list):
    return [H.equivalence[H.equivalence[:, 0] == kk, 1][0] for kk in k]
  else: return np.min(H.equivalence[H.equivalence[:, 0] == k, 1])

def get_k_from_equivalence(r, H):
  if isinstance(r, list):
    return [H.equivalence[H.equivalence[:, 1] == r, 1][0] for rr in r]
  else: return np.min(H.equivalence[H.equivalence[:, 1] == r, 1])

def get_r_from_X_diag(K, H, Z, R, nodes, **kwargs):
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  r = []
  for k in K:
    labels = cut_tree(H, k).ravel()
    dR = adj2df(R[:nodes, :])
    dR = dR.loc[(dR.weight != 0)]
    dR["id"] = labels
    minus_one_Dc(dR, False)
    aesthetic_ids(dR)
    ##
    den_order = np.array(
      dendrogram(Z, no_plot=True)["ivl"]
    ).astype(int)
    RR = df2adj(dR, var="id")[:, den_order][den_order, :]
    dR = adj2df(RR)
    dR["id"] = dR.weight
    dR = dR.loc[(dR.id != 0)]
    ##
    unique_labels = np.unique(dR.id)
    len_unique_labels = 0
    dR = dR.loc[dR.id != -1]
    for label in unique_labels:
      if label == -1: continue
      dr_down = dR.source.loc[(dR.id == label) & (dR.target < dR.source)]
      dr_up = dR.target.loc[(dR.id == label) & (dR.source < dR.target)]
      len_between_up_down = len(set(dr_up).intersection(set(dr_down)))
      if len_between_up_down > 0:
        len_unique_labels += 1
      else:
        dR = dR.loc[dR.id != label]
    nodes_fair_communities = set(dR.source).intersection(set(dR.target))
    r.append(nodes - len(nodes_fair_communities) + len_between_up_down)
  return r

def get_r_from_modularity(k, H, Z, R, nodes, undirected=False):
  from collections import Counter
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  if not undirected:
    RR = R.copy()
    RR[RR != 0] = 1
    RR = RR[:nodes, :]
    RR = adj2df(RR)
    RR = RR.loc[RR.weight != 0]
    RR["id"] = cut_tree(H, k).ravel()
  else:
    RR = np.triu(R)
    nonzero = RR != 0
    nonx, nony = np.where(nonzero)
    RR = pd.DataFrame(
      {
        "source" : list(nonx) + list(nony),
        "target" : list(nony) + list(nonx),
        "weight" : list(R[nonx, nony]) * 2
      }
    )
    RR["id"] = np.tile(cut_tree(H, k).ravel(), 2)
  minus_one_Dc(RR, undirected=undirected)
  aesthetic_ids(RR)
  RR2 = df2adj(RR, var="id")
  RR2[RR2 == -1] = 0
  RR2[RR2 != 0] = 1
  RR = df2adj(RR)
  RR = RR * RR2
  #
  den_order = np.array(dendrogram(Z, no_plot=True)["ivl"]).astype(int)
  RR = RR[den_order, :][:, den_order]
  D = np.zeros(nodes - 1)
  for i in np.arange(nodes - 1, 0, -1):
    partition = cut_tree(Z, i).ravel()[den_order]
    number_nodes = dict(Counter(partition))
    where_nodes = {k : np.where(partition == k)[0] for k in number_nodes.keys()}
    d = np.array([(number_nodes[k] / nodes) * np.nansum(RR[where_nodes[k], :][:, where_nodes[k]]) / (number_nodes[k] * (number_nodes[k] - 1)) for k in number_nodes.keys()])
    D[i-1] = np.nansum(d)
  return np.argmax(D)

get_r_from = {
  "modularity" : get_r_from_modularity,
  "X_diag" : get_r_from_X_diag
}

def get_r_from_X_diag_2(k, H, Z, R, nodes):
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  labels = cut_tree(H, k).ravel()
  dR = adj2df(R[:nodes, :])
  dR = dR.loc[(dR.weight != 0)]
  dR["id"] = labels
  minus_one_Dc(dR)
  aesthetic_ids(dR)
  ##
  den_order = np.array(
    dendrogram(Z)["ivl"]
  ).astype(int)
  RR = df2adj(dR, var="id")[:, den_order][den_order, :]
  dR = adj2df(RR)
  dR["id"] = dR.weight
  dR = dR.loc[(dR.id != 0)]
  ##
  unique_labels = np.unique(dR.id)
  len_unique_labels = 0
  fair_nodes = 0
  for label in unique_labels:
    if label == -1: continue
    dr_down = dR.source.loc[(dR.id == label) & (dR.target < dR.source)]
    dr_up = dR.target.loc[(dR.id == label) & (dR.source < dR.target)]
    len_between_up_down = len(set(dr_down).intersection(set(dr_up)))
    if len_between_up_down > 0:
      len_unique_labels += 1
      fair_nodes += len_between_up_down
  return nodes - fair_nodes + len_unique_labels

def stats_maxsize_linkcom(data):
    subdata = data.groupby(["nodes"])["size"].max()
    mean = subdata.mean()
    std = subdata.std()
    return mean, std

def tree_dominant_nodes(data, labels):
  catch_nodes = []
  subdata = data.groupby(["nodes"]).max()
  max_size = subdata["size"]
  nodes = subdata.index
  for i, nd in enumerate(nodes):
    ids = data["ids"].loc[data["nodes"] == nd].to_numpy()
    if -1 not in ids: continue
    x = data.loc[
      (data["nodes"] == nd) & (data["ids"] == -1),
      "size"
    ].iloc[0]
    if x == max_size.iloc[i]:
      y = np.where(labels == nd)[0][0]
      catch_nodes.append(y)
  return catch_nodes

def NMI_single(labels, H, on=True, **kwargs):
  if on:
    K = []
    if "K" in kwargs.keys():
      for kk in kwargs["K"]:
        BH = H.BH[0]
        k = BH["NAC"].loc[BH["K"] == kk].to_numpy()
        K.append(k[0])
    else:
      K.append(np.unique(labels).shape[0])
    from scipy.cluster.hierarchy import cut_tree
    for k in K:
      h_ids = cut_tree(
        H.Z,
        n_clusters=k
      ).reshape(-1)
      from sklearn.metrics import normalized_mutual_info_score
      nmi = normalized_mutual_info_score(labels, h_ids, average_method="max")
      print("NMI({}): {}".format(k, nmi))

def NMI_average(gt, K, R, WSBM, on=True):
  if on:
    from sklearn.metrics import normalized_mutual_info_score
    pred = WSBM.labels.loc[
      (WSBM.labels["K"] == K) &
      (WSBM.labels["R"] == R),
      "labels"
    ].to_numpy()
    nmi = normalized_mutual_info_score(gt, pred)
    print("K: {}\tR: {}\tNMI: {}".format(K, R, nmi))
    return nmi

def NMI_label(gt, pred, on=True):
  if on:
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(gt, pred, average_method="max")
    print("NMI: {}".format(nmi))
    return nmi

def AD_NMI_label(gt, pred, on=True):
  if on:
    if np.sum(np.isnan(pred)) > 0: nmi = np.nan
    elif len(np.unique(pred)) == 1: nmi = np.nan
    else:
      p = pred.copy()
      none = np.sum(p == -1)
      p[p == -1] = np.arange(np.max(p) + 1, np.max(p) + 1 + none)
      nmi = adjusted_mutual_info_score(gt, p, average_method="max")

    print("ADNMI: {}".format(nmi))
    return nmi

def AD_NMI_overlap(gt, pred, on=True):
  if on:
    if np.sum(np.isnan(pred)) > 0: nmi = np.nan
    elif len(np.unique(pred)) == 1: nmi = np.nan
    else:
      predx = pred.copy()
      none = np.sum(predx == -1)
      predx[predx == -1] = np.arange(np.max(predx) + 1, np.max(predx) + 1 + none)
      nmi = adjusted_mutual_info_score(gt, predx, average_method="max")
    print("ADNMI: {:.4f}".format(nmi))
    return nmi
  else:
    return -1

def stirling(n):
    return math.sqrt(2*math.pi*n)*(n/math.e)**n

def modAD_NMI_overlap(gt : dict, pred : dict, on=True):
  ngt = len(gt)
  npred = len(pred)
  NUV = np.zeros((ngt, npred))
  for i, (kgt, vgt) in enumerate(gt.items()):
    for j, (kpred, vpred) in enumerate(pred.items()):
      NUV[i,j] = len(set(vgt).intersection(set(vpred)))
  Neff = np.sum(NUV)
  NU = np.sum(NUV, axis=1).ravel()
  NV = np.sum(NUV, axis=0).ravel()

  MI = 0
  
  for i in np.arange(ngt):
    for j in np.arange(npred):
      if NUV[i,j] > 0:
        MI += (NUV[i, j] / Neff) * np.log(Neff * NUV[i,j]/ (NU[i] * NV[j]))

  MIrand = 0

  for i in np.arange(ngt):
    ai = NU[i]
    for j in np.arange(npred):
      bj = NV[j]
      lw = np.maximum(0, ai + bj - Neff)
      up = np.minimum(ai, bj)
      hy = hypergeom(Neff, bj, ai)
      for nij in np.arange(lw, up):
          if nij > 0:
            MIrand += (nij/Neff) * np.log((Neff * nij)/(ai*bj)) * hy.pmf(nij)
  
  HU = np.sum([-(nu/Neff) * np.log(nu / Neff) for nu in NU])
  HV = np.sum([-(nv/Neff) * np.log(nv / Neff) for nv in NV])

  score = (MI - MIrand) / (np.maximum(HU, HV) - MIrand)
  print("modADNMI: {:.4f}".format(score))
  return score 

def save_class(
  CLASS, pickle_path, class_name="duck", on=True, **kwargs
):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  if exists(path): remove(path)
  if on:
    with open(path, "wb") as f:
      pk.dump(CLASS, f)

def read_class(pickle_path, class_name="duck", **kwargs):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  C = 0
  print(path)
  if isfile(path) and stat(path).st_size > 100:
    with open(path, "rb") as f:
      C =  pk.load(f)
  else: print(f"\nFile {path} does not exist\n")
  return C

def column_normalize(A):
  if np.sum(np.isnan(A)) > 0:
    raise ValueError("\nColumn normalied does not accept nan. Use instead column_normalize_nan.\n")
  C = A.sum(axis = 0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def column_normalize_nan(A):
  C = np.nansum(A, axis=0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def match(a, b):
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])
    
def sort_by_size(ids, nodes):
  # Define location memory ---
  c = 0
  # Define new id ----
  nids = np.zeros(nodes)
  # Find membership frequency ----
  from collections import Counter
  f = Counter(ids)
  f = dict(f)
  f = sort_dict_value(f)
  for key in f:
    w = np.where(ids == key)[0]
    lw = len(w)
    nids[c:(lw + c)] = w
    c += lw
  return nids.astype(int), f

def sort_dict_value(counter : dict):
  f = {
    k: v for k, v in sorted(
      counter.items(), key=lambda item: item[1],
      reverse=True
    )
  }
  return f

def invert_dict_single(f: dict):
  return {v : k for k, v in f.items()}

def invert_dict_multiple(f : dict):
  ff = {}
  for key in f.keys():
    for val in f[key]:
      if val not in ff.keys(): ff[val] = [key]
      else: ff[val] = ff[val] + [key]
  for key in ff.keys(): ff[key] = list(np.unique(ff[key]))
  return ff

# def membership2ids(Cr, dA):
#   from collections import Counter
#   cr_skim = skim_partition(Cr)
#   skm = np.unique(cr_skim)
#   skm = skm[skm != -1]
#   nodecom_2_id = {}
#   for r in skm:
#     r_nodes = np.where(cr_skim == r)[0]
#     if len(r_nodes) == 0: continue
#     dr = dA.loc[
#       (np.isin(dA.source, r_nodes)) &
#       (np.isin(dA.target, r_nodes))
#     ]
#     id_r_counter = dict(Counter(dr.id))
#     id_r_counter = sort_dict_value(id_r_counter).keys()
#     nodecom_2_id[r] = list(id_r_counter)[0]
#   return nodecom_2_id

def membership2ids(Cr, dA):
  skimCr = skim_partition(Cr)
  # uCr = np.unique(Cr)
  uCr = np.unique(skimCr[skimCr != -1])
  nodecom_2_id = {
    ur : [] for ur in uCr
  }
  for ur in uCr:
    ur_nodes = np.where(Cr == ur)[0]
    if len(ur_nodes) == 0: continue
    dur = np.unique(dA.id.loc[np.isin(dA.source, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
    dur = np.unique(dA.id.loc[np.isin(dA.target, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
  for key in nodecom_2_id:
    nodecom_2_id[key] = list(np.unique(nodecom_2_id[key]))
  return nodecom_2_id

def condense_madtrix(A):
  n = A.shape[0] * (A.shape[0] - 1) / 2
  cma = np.zeros(int(n))
  t = 0
  for i in np.arange(A.shape[0] - 1):
    for j in np.arange(i + 1, A.shape[0]):
      cma[t] =  A[i, j]
      t += 1
  return cma

def df2adj(dA, var="weight"):
  m = np.max(dA["source"].to_numpy()) + 1
  n = np.max(dA["target"].to_numpy()) + 1
  A = np.zeros((m.astype(int), n.astype(int)))
  A[
    dA["source"].to_numpy().astype(int),
    dA["target"].to_numpy().astype(int)
  ] = dA[var].to_numpy()
  return A

def adj2df(A):
  src = np.repeat(
    np.arange(A.shape[0]),
    A.shape[1]
  )
  tgt = np.tile(
    np.arange(A.shape[1]),
    A.shape[0]
  )
  dA = pd.DataFrame(
    {
      "source" : src,
      "target" : tgt,
      "weight" : A.reshape(-1)
    }
  )
  return dA

def omega_index_format(node_partition, noc_covers : dict, node_labels):
  rev = reverse_partition(node_partition, node_labels)
  nocs2parition(rev, noc_covers)
  return rev

def reverse_cover(cover: dict, labels):
  cover_indices = set()
  for k, v in cover.items():
    cover_indices = cover_indices.union(set(v))
  rev = {k : [] for k in cover_indices}
  for k, v in cover.items():
    for vv in v:
      rev[vv].append(labels[k])
  return rev

def omega_index(cover_1 : dict, cover_2 : dict):
  if len(cover_1) == 1 and len(cover_2) == 1:
    omega = np.nan
  else:
    omega = Omega(cover_1, cover_2).omega_score
  print(f"Omega: {omega:.4f}")
  return omega
    
