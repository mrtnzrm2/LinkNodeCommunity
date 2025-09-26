import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
import seaborn as sns

from LinkNodeCommunity.core import nocs

pd.options.mode.chained_assignment = None  # default='warn'


def hex_to_rgb(hex_value: str):
  """
  Convert a hex color string to an RGB tuple.

  Parameters
  ----------
  hex_value : str
      The hex color string (e.g., "#RRGGBB").
  Returns
  -------
  tuple
      A tuple representing the RGB color (R, G, B).
      Each component is a float in the range [0, 1].
  """
  return tuple(int(hex_value.lstrip("#")[i:i+2], 16)/255.0 for i in (0, 2, 4))

def edgelist_from_graph(G: nx.DiGraph | nx.Graph, sort: bool = False):
  """
  Convert a NetworkX graph to an edge list DataFrame with columns ['source', 'target', 'weight'].
  Optionally sort the DataFrame by 'source' (first) and 'target' (second).

  Parameters
  ----------
  G : nx.DiGraph or nx.Graph
    The input NetworkX graph.
  sort : bool, optional
    If True, sort the DataFrame by 'source' and then 'target'. Default is False.

  Returns
  -------
  pd.DataFrame
    Edge list DataFrame with columns ['source', 'target', 'weight'].
  """
  edgelist = [
      {"source": u, "target": v, "weight": data.get("weight", 1.0)}
      for u, v, data in G.edges(data=True)
    ]
  edgelist = pd.DataFrame(edgelist)
  if sort:
    edgelist = edgelist.sort_values(["source", "target"]).reset_index(drop=True)
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

def cover_map_to_partition(partition: dict, single_node_to_covers_map: dict):
  """
  Update the partition dictionary so that each cover (community label) includes
  the string representation of each noc (node of cover) from the mapping.

  Parameters
  ----------
  partition : dict
    Dictionary mapping community labels to lists of node labels.
  single_node_to_covers_map : dict
    Dictionary mapping node labels to lists of covers (community labels).
  """
  for noc, covers in single_node_to_covers_map.items():
    for cover in covers:
      label = str(noc)
      if label not in partition.setdefault(cover, []):
        partition[cover].append(label)


def reverse_partition(partition: npt.ArrayLike, labels: npt.ArrayLike):
  """
  Given a partition (community labels for each node) and corresponding labels,
  return a dictionary mapping each community label to the list of node labels in that community.
  Ignores nodes with label -1.

  Parameters
  ----------
  partition : npt.ArrayLike
    Array of community labels for each node.
  labels : npt.ArrayLike
    Array of node labels.

  Returns
  -------
  dict
    Dictionary mapping community label to list of node labels.
  """
  partition = np.asarray(partition)
  labels = np.asarray(labels)
  mask = partition != -1
  communities = np.unique(partition[mask])
  return {int(r): list(labels[partition == r]) for r in communities}


def format_partition_for_omega_index(partition : npt.ArrayLike, single_node_to_covers_map : dict, labels : npt.ArrayLike| None=None):
  '''
  Format a partition and single-node-to-covers map for omega index calculation.
  '''
  if labels is None:
    labels = np.arange(len(partition))
  rev = reverse_partition(partition, labels)
  cover_map_to_partition(rev, single_node_to_covers_map)
  return rev

def fast_cut_tree(H : npt.NDArray, n_clusters=None, height=None):
  '''
  Lightweight replacement for scipy.cluster.hierarchy.cut_tree supporting a
  single ``n_clusters`` *or* ``height`` cut.

  Parameters
  ----------
  H : npt.ArrayLike
      The hierarchical clustering encoded as a linkage matrix.
  n_clusters : int, optional
      The number of clusters to form. Must be mutually exclusive with `height`.
  height : float, optional
      The height to cut the dendrogram. Must be mutually exclusive with `n_clusters`.
  '''

  H = np.asarray(H)
  if H.ndim != 2 or H.shape[1] != 4:
    raise ValueError("H must be a linkage matrix with shape (n-1, 4).")

  has_clusters = n_clusters is not None
  has_height = height is not None
  if has_clusters == has_height:
    raise ValueError("Specify exactly one of n_clusters or height.")

  n_leaves = H.shape[0] + 1
  active_clusters = set(range(n_leaves))
  cluster_members: dict[int, list[int]] = {}

  if has_height:
    threshold = float(height)
    for i, (a, b, dist, _) in enumerate(H):
      if dist >= threshold:
        break
      a = int(a)
      b = int(b)
      members_a = [a] if a < n_leaves else cluster_members.pop(a)
      members_b = [b] if b < n_leaves else cluster_members.pop(b)
      new_idx = n_leaves + i
      cluster_members[new_idx] = members_a + members_b
      active_clusters.discard(a)
      active_clusters.discard(b)
      active_clusters.add(new_idx)
  else:
    target = int(n_clusters)
    if target < 1 or target > n_leaves:
      raise ValueError("n_clusters must be in [1, n_leaves].")
    for i, (a, b, dist, _) in enumerate(H):
      if len(active_clusters) <= target:
        break
      a = int(a)
      b = int(b)
      members_a = [a] if a < n_leaves else cluster_members.pop(a)
      members_b = [b] if b < n_leaves else cluster_members.pop(b)
      new_idx = n_leaves + i
      cluster_members[new_idx] = members_a + members_b
      active_clusters.discard(a)
      active_clusters.discard(b)
      active_clusters.add(new_idx)

  partition = np.zeros(n_leaves, dtype=np.int64)
  for label, cid in enumerate(sorted(active_clusters)):
    members = [cid] if cid < n_leaves else cluster_members[cid]
    partition[members] = label

  return partition

def linear_partition(partition : npt.ArrayLike):
  ''' 
  Renumber the communities linearly. From 0 to number of communities - 1, except entries that are -1 (which remain -1).

  Parameters
  ----------
  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------
  npt.NDArray
      Renumbered partition with labels from 0 to number of communities - 1. Entries with -1 remain -1.
  '''
  
  par = np.asarray(partition).copy()
  mask = par != -1
  unique_labels = np.unique(par[mask])
  label_map = {label: i for i, label in enumerate(unique_labels)}
  new_partition = np.array([label_map[x] if x != -1 else -1 for x in par])
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
    print("Warning: more than one link community level with maximum D")
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
    print("Warning: more than one link community level with maximum S")
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


def linkDc(df: pd.DataFrame, id, undirected=False):
  """
  Compute the density (Dc) of a link community specified by 'id' in an edge list DataFrame.

  Parameters
  ----------
  df : pd.DataFrame
    DataFrame containing edge list with columns ['source', 'target', 'id'].
  id : int or str
    The link community membership label to evaluate.
  undirected : bool, optional
    If True, treats the graph as undirected and considers only edges where source > target.
    If False, treats the graph as directed. Default is False.

  Returns
  -------
  float
    The density of the link community. Returns 0 for trivial communities (single node or not enough edges).
  """
  # Filter edges belonging to the specified community
  if undirected:
    df2 = df[(df["id"] == id) & (df["source"] > df["target"])]
  else:
    df2 = df[df["id"] == id]

  nodes = set(df2["source"]).union(df2["target"])
  m = len(df2)
  n = len(nodes)

  if n <= 1 or m < n:
    return 0.0

  if undirected:
    denom = n * (n - 1) / 2. - n + 1.
    return (m - n + 1) / denom if denom > 0 else 0.0
  else:
    return (m - n + 1) / (n - 1) ** 2

def linkcommunity_collapsed_partition(df: pd.DataFrame, undirected: bool = False):
  """
  Collapse trivial link communities in an edge list DataFrame by assigning their
  membership 'id' to -1. A trivial community is one with density <= 0.

  Parameters
  ----------
  df : pd.DataFrame
    Edge list DataFrame with columns ['source', 'target', 'id'] representing link community membership.
  undirected : bool, optional
    If True, treats the graph as undirected for density calculation. Default is False.

  Modifies
  --------
  df : pd.DataFrame
    Updates the 'id' column in-place, setting trivial communities to -1.
  """
  for cid in np.unique(df["id"]):
    if linkDc(df, cid, undirected=undirected) <= 0:
      df.loc[df["id"] == cid, "id"] = -1

def linkcommunity_linear_partition(df: pd.DataFrame, offset: int = 0):
  """
  Renumber non-trivial link community memberships in the 'id' column of an edge list DataFrame.
  Non-trivial communities (id != -1) are mapped to consecutive integers from offset to C + offset - 1,
  where C is the number of non-trivial link communities. Trivial communities (id == -1)
  remain unchanged.

  Parameters
  ----------
  df : pd.DataFrame
    Edge list DataFrame with a link community membership column 'id'.

  Modifies
  --------
  df : pd.DataFrame
    Updates the 'id' column in-place, renumbering non-trivial communities.
  """
  ids = np.unique(df["id"])
  non_trivial = ids[ids != -1]
  mapping = {id_: i + offset for i, id_ in enumerate(non_trivial)}
  # Keep -1 as is
  mapping[-1] = -1 if -1 in ids else None
  df["id"] = df["id"].map(lambda x: mapping.get(x, x)).astype(int)


def adjacency_to_edgelist(A: np.ndarray) -> pd.DataFrame:
  """
  Converts an adjacency matrix to an edge list DataFrame.

  Parameters
  ----------
  A : np.ndarray
    A 2D numpy array representing the adjacency matrix of a graph,
    where A[i, j] indicates the weight of the edge from node i to node j.

  Returns
  -------
  pd.DataFrame
    A DataFrame with columns:
      - 'source': Source node indices.
      - 'target': Target node indices.
      - 'weight': Edge weights corresponding to each (source, target) pair.

  Notes
  -----
  - The resulting edge list includes all possible edges, including those with zero weight.
  - The function assumes that the adjacency matrix is square (n x n) for n nodes.

  Examples
  --------
  >>> import numpy as np
  >>> import pandas as pd
  >>> A = np.array([[0, 1], [2, 0]])
  >>> adjacencty_to_edgelist(A)
     source  target  weight
  0       0       0       0
  1       0       1       1
  2       1       0       2
  3       1       1       0
  """
  src = np.repeat(
    np.arange(A.shape[0]),
    A.shape[1]
  )
  tgt = np.tile(
    np.arange(A.shape[1]),
    A.shape[0]
  )
  df = pd.DataFrame(
    {
      "source" : src,
      "target" : tgt,
      "weight" : A.reshape(-1)
    }
  )
  return df


def edgelist_to_adjacency(df: pd.DataFrame, weight: str = "weight") -> np.ndarray:
  """
  Convert an edge list DataFrame to an adjacency matrix.

  Parameters
  ----------
  df : pd.DataFrame
    Edge list DataFrame with columns ['source', 'target'] and optionally a weight column.
  weight : str, optional
    Name of the column to use for edge weights. Default is "weight".

  Returns
  -------
  np.ndarray
    Adjacency matrix where entry (i, j) is the weight of the edge from i to j.
    If no edge exists, the entry is 0.
  """
  sources = df["source"].astype(int).to_numpy()
  targets = df["target"].astype(int).to_numpy()
  weights = df[weight].to_numpy()
  num_nodes = max(sources.max(), targets.max()) + 1
  adj = np.zeros((num_nodes, num_nodes), dtype=weights.dtype)
  adj[sources, targets] = weights
  return adj