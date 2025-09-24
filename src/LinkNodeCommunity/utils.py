import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'

def edgelist_from_graph(G : nx.DiGraph | nx.Graph):
  """
  Convert a NetworkX graph to an edge list DataFrame with columns ['source', 'target', 'weight'].
  """
  edgelist = [
      {"source": u, "target": v, "weight": data.get("weight", 1.0)}
      for u, v, data in G.edges(data=True)
    ]
  edgelist = pd.DataFrame(edgelist)
  return edgelist

def generate_cmap_from_partition(partition : npt.ArrayLike, trivial="-1", cmap="hls", seed=None, numeric=True):
  '''
  Create a dictionary from a list of community memberships K, where each unique
  element in K is associated with a color from palette cmap. Notice, K is expected
  to have numeric strings, but it also can handle other formats. The trivial element '-1'
  is especial and it is linked to the gray color.

  Parameters
  ----------
  partition : npt.ArrayLike
      Array of community memberships.
  trivial : str, optional
      The trivial community label. Default is "-1".
  cmap : str, optional
      The colormap to use. Default is "hls".
  seed : int, optional
      Random seed for reproducibility. Default is None.
  numeric : bool, optional
      Whether to treat community labels as numeric. Default is True.

  Returns
  -------
  cm : dict
      Dictionary mapping community labels to colors.
  '''
  from matplotlib.colors import to_hex
  if numeric:
    unique_labels = np.sort(np.unique(partition.astype(int))).astype(str)
  else:
    unique_labels = np.sort(np.unique(partition)).astype(str)

  if np.isin(trivial, unique_labels):
      if seed is None:
        cm = list(sns.color_palette(cmap, unique_labels.shape[0]-1))
      else:
        print(f"Seed: {seed}")
        np.random.seed(seed)
        n = len(unique_labels)
        cm = list(np.array(sns.color_palette(cmap, unique_labels.shape[0]-1))[np.random.permutation(np.arange(n-1))])
      cm = [to_hex((0.5, 0.5, 0.5))] + cm
  else:
    if seed is None:
      cm = list(sns.color_palette(cmap, unique_labels.shape[0]))
    else:
      print(f"Seed: {seed}")
      np.random.seed(seed)
      n = len(unique_labels)
      cm = list(np.array(sns.color_palette(cmap, unique_labels.shape[0]))[np.random.permutation(np.arange(n))])

  cm = {u: to_hex(c) for u, c in zip(unique_labels, cm)}
  return cm

def consecutive_differences(d : npt.ArrayLike) -> npt.ArrayLike:
  """
  Compute difference between consecutive elements of a 1D array.
  """
  return np.array([d[i+1] - d[i] for i in np.arange(d.shape[0]-1)])


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
      Renumbered partition with labels from 0 to number of communities - 1.
  '''
  
  par = np.asarray(partition).copy()
  unique_labels = np.unique(par)
  label_map = {label: i for i, label in enumerate(unique_labels)}
  new_partition = np.array([label_map[x] for x in par])
  return new_partition

def collapsed_partition(partition : npt.ArrayLike):
  '''
  Renumber the communities linearly. From 0 to number of communities - 1. Singletons are replaced by -1.

  Parameters
  ----------
  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------
  npt.NDArray
      Renumbered partition with labels from 0 to number of communities - 1. Singletons are replaced by -1.
  '''

  from collections import Counter
  par = np.asarray(partition).copy()
  counts = Counter(par)
  # Set singleton communities to -1
  for label, count in counts.items():
    if count == 1:
      par[par == label] = -1
  # Renumber remaining communities from 0, keep -1 as is
  unique_labels = np.unique(par[par != -1])
  label_map = {label: i for i, label in enumerate(unique_labels)}
  new_partition = np.array([label_map[x] if x != -1 else -1 for x in par])
  return new_partition

def linknode_equivalence_partition(score, link_stats : pd.DataFrame, linknode_equivalence : npt.NDArray):
  
  """
  Compute the number of link communities (K) and the corresponding number of node
  communities (R) from the provided link statistics DataFrame using the
  link-node equivalence approach. The selection is based on the maximum value of
  the specified score. It also returns the height at which the maximum occurs.

  Parameters
  ----------
  score : str
      The score to use for selecting the maximum (either "D" or "S").
  link_stats : pd.DataFrame
      DataFrame containing link community statistics with columns 'K', 'S', 'D', and 'height'.

  Returns
  -------
  tuple
      - number_link_communities (int or np.ndarray): Number of link communities at maximum score.
      - number_node_communities (int or np.ndarray): Corresponding number of node communities.
      - height_at_maximum (float or np.ndarray): Height at which the maximum score occurs.
  """

  if score not in {"D", "S"}:
    raise ValueError("Unexpected score: must be 'D' or 'S'")

  get_max_func = get_number_link_communities_from_maxD if score == "D" else get_number_link_communities_from_maxS
  number_link_communities, height_at_maximum = get_max_func(link_stats)
  number_node_communities = get_number_node_communities_from_linknode_equivalence(number_link_communities, linknode_equivalence)

  # Ensure all outputs are numpy arrays of at least 1D, unless they are scalars
  def to_array_or_scalar(x):
    if isinstance(x, np.ndarray):
      if x.size == 1:
        return x.item()
      return x
    elif isinstance(x, (list, tuple)):
      if len(x) == 1:
        return x[0]
      return np.array(x)
    else:
      return x

  return (
    to_array_or_scalar(number_link_communities),
    to_array_or_scalar(number_node_communities),
    to_array_or_scalar(height_at_maximum)
  )

def is_valid_linkage_matrix(Z: npt.ArrayLike) -> bool:
    """
    Validate if a matrix Z is a valid linkage matrix in SciPy format.

    Parameters
    ----------
    Z : npt.ArrayLike
      The matrix to validate.

    Returns
    -------
    bool
      True if Z is a valid linkage matrix, False otherwise.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 4:
      return False
    n = Z.shape[0] + 1
    # Check that the first two columns are valid indices
    valid_indices = np.arange(2 * n - 1)
    if not np.all(np.isin(Z[:, 0], valid_indices)) or not np.all(np.isin(Z[:, 1], valid_indices)):
      return False
    # Check that distances are non-negative and increasing
    if np.any(Z[:, 2] < 0):
      return False
    if not np.all(np.diff(Z[:, 2]) >= 0):
      return False
    # Check that the last column (number of original observations in the cluster) is >= 2
    if np.any(Z[:, 3] < 2):
      return False
    return True


def cut_tree_with_validation(Z : npt.NDArray, number_communities : int = None, height : float = None):
  if not is_valid_linkage_matrix(Z):
    raise ValueError("Z is not a valid linkage matrix.")
  if number_communities is None and height is None:
    raise ValueError("Either number_communities or height must be provided.")
  if number_communities is not None and height is not None:
    raise ValueError("Only one of number_communities or height should be provided.")
  if height is not None and number_communities is None:
    return fast_cut_tree(Z, height=height)
  else:
    return fast_cut_tree(Z, n_clusters=number_communities)
  

def get_number_link_communities_from_maxD(link_stats : pd.DataFrame):
  """
  Get the number of link communities (K) and the height at which the maximum D
  (average link density) occurs from the provided statistics DataFrame.
  NOTE: If multiple maxima exist, the last one is returned.

  Parameters
  ----------
  link_stats : pd.DataFrame
      DataFrame containing link community statistics with columns 'K', 'D', and 'height'.
  """
  number_link_communities = link_stats["K"].loc[
    link_stats["D"] == np.nanmax(link_stats["D"])
  ]
  height_at_maxD = link_stats["height"].loc[
    link_stats["D"] == np.nanmax(link_stats["D"])
  ]
  if number_link_communities.shape[0] > 1:
    print(">>> Warning: more than one k")
  return int(number_link_communities.iloc[-1]), float(height_at_maxD.iloc[-1])

def   get_number_link_communities_from_maxS(link_stats : pd.DataFrame):
  """
  Get the number of link communities (K) and the height at which the maximum S
  (loop entropy) occurs from the provided statistics DataFrame.
  NOTE: If multiple maxima exist, the last one is returned.

  Parameters
  ----------
  link_stats : pd.DataFrame
      DataFrame containing link community statistics with columns 'K', 'S', and 'height'.
  """
  number_link_communities = link_stats["K"].loc[
    link_stats["S"] == np.nanmax(link_stats["S"])
  ]
  height_at_maxS = link_stats["height"].loc[
    link_stats["S"] == np.nanmax(link_stats["S"])
  ]
  if number_link_communities.shape[0] > 1:
    print(">>> Warning: more than one k")
  return int(number_link_communities.iloc[-1]), float(height_at_maxS.iloc[-1])


def get_number_node_communities_from_linknode_equivalence(number_link_communities: npt.ArrayLike| int, linknode_equivalence : npt.NDArray):
  """
  Get the number of node communities (r) corresponding to a given number of link
  communities (K) using the provided link-node equivalence array.
  NOTE: If multiple r correspond to the same K, the minimum r is returned.

  Parameters
  ----------
  number_link_communities : list of int or int
      The number of link communities (K).
  linknode_equivalence : npt.NDArray
      The link-node equivalence array.
  """
  if isinstance(number_link_communities, (list, tuple, np.ndarray)):
    return [linknode_equivalence[linknode_equivalence[:, 0] == kk, 1][0] for kk in number_link_communities]
  else: return np.min(linknode_equivalence[linknode_equivalence[:, 0] == number_link_communities, 1])


def get_number_link_communities_from_linknode_equivalence(number_node_communities : npt.ArrayLike| int, linknode_equivalence : npt.NDArray):
  """
  Get the number of link communities (K) corresponding to a given number of node
  communities (r) using the provided link-node equivalence array.
  NOTE: If multiple K correspond to the same r, the minimum K is returned.

  Parameters
  ----------
  number_node_communities : int or list of int
      The number of node communities (r).
  linknode_equivalence : npt.NDArray
      The link-node equivalence array.
  """
  if isinstance(number_node_communities, (list, tuple, np.ndarray)):
    return [linknode_equivalence[linknode_equivalence[:, 1] == rr, 0][0] for rr in number_node_communities]
  else: return np.min(linknode_equivalence[linknode_equivalence[:, 1] == number_node_communities, 0])


def match(a : npt.ArrayLike, b : npt.ArrayLike):
    """
    Match elements of array `a` to their indices in array `b`.

    Parameters
    ----------
    a : npt.ArrayLike
        The array whose elements are to be matched.
    b : npt.ArrayLike
        The array to match elements against.
    """
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])