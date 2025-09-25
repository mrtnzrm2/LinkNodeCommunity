"""
Module: linknode
Author: Jorge S. Martinez Armas

Overview:
---------
NOCFinder assigns isolated/single nodes (partition == -1) to one or more
non‑trivial communities based on distances derived from node–node similarity
matrices. For directed graphs it considers two perspectives (source/outgoing
and target/incoming) and merges evidence from both. When a single node is
assigned to multiple communities, it is considered a NOC (Node with Overlapping
Community membership).

Parameters (NOCFinder):
-----------------------
NOCFinder(G, node_partition, undirected=False,
          similarity_index="hellinger_similarity",
          tie_policy="include_equal", eps=0.0, node_order=None)

- G (nx.Graph | nx.DiGraph): Input graph. Nodes may be any hashables.
- node_partition (array-like, len N): Community id per node; use -1 for singles.
- undirected (bool): Whether upstream steps treat edges as undirected.
- similarity_index (str): One of {hellinger_similarity, cosine_similarity,
  pearson_correlation, weighted_jaccard, jaccard_probability,
  tanimoto_coefficient}. Controls similarity→distance transform.
- tie_policy (str): One of {cluster_only, include_equal, include_equal_same_cluster,
  deterministic}. Governs how multiple close communities are selected.
- eps (float): Tolerance for equality comparisons in tie handling.
- node_order (array-like | None): Explicit node order aligning G, partition,
  and similarity matrices. Defaults to sorted G nodes if None.



Goals:
------
- Fill community memberships for nodes with partition == -1.
- Allow overlapping assignments via configurable tie policies.
- Provide a similarity‑proxy score per assigned cover.
- Support directed/undirected graphs and several similarity indices.

Notes:
------
- Ordering/alignment: Builds or validates a node order and maps node ids to
  contiguous indices so that partition and similarity matrices align.
- Distance transforms: Similarities are converted to distances differently for
  correlation/cosine/hellinger (sqrt(2(1-s))) vs. Jaccard/Tanimoto (1-s).
- Outputs: node_cover_partition (hard assignments where unique),
  single_node_cover_map (overlaps), single_nodes_cover_scores (scores).
- This class may be slow for large graphs, as it relies on Python for-loops
  and does not use a low-level backend.
- Tie handling: If multiple candidate communities are close, a COST matrix of
  |Δ distance| is hierarchically clustered and a tie policy selects covers.
- Validation: Shapes, accepted parameters, and node_order content are checked.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import networkx as nx
import pandas as pd
import numpy.typing as npt

# Node Overlapping Communities ----
class NOCFinder:
  def __init__(
    self,
    G : nx.DiGraph | nx.Graph,
    node_partition : npt.ArrayLike,
    undirected: bool = False,
    labels : npt.ArrayLike | None = None,
    similarity_index: str = "hellinger_similarity",
    tie_policy: str = "include_equal",
    eps: float = 0.0,
    node_order: npt.ArrayLike | None = None,
    max_workers: int | None = None,
  ):
    """
    Assigns single/isolated nodes (partition == -1) to one or more
    non-trivial communities using distances derived from node-node
    similarity matrices. Nodes assigned to multiple communities are
    treated as NOCs (overlapping memberships).

    NOTE: node_partition must have -1 for single nodes.

    NOTE: No low-level backend; may be slow for large graphs.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Input graph. Nodes may be any hashables.
    node_partition : array-like, len N
        Community id per node; use -1 for singles. Must contain -1 for single nodes.
    undirected : bool, optional
        Whether upstream steps treat edges as undirected. Default is False.
    labels : array-like | None, optional
        Node labels for interpretation of results. Default is None.
    similarity_index : str, optional
        One of {hellinger_similarity, cosine_similarity,
        pearson_correlation, weighted_jaccard, jaccard_probability,
        tanimoto_coefficient}. Controls similarity→distance transform.
        Default is "hellinger_similarity".
    tie_policy : str, optional
        One of {cluster_only, include_equal, include_equal_same_cluster, deterministic}.
        Controls tie-breaking behavior. Default is "include_equal".
    eps : float, optional
        Small value to avoid division by zero in similarity calculations. Default is 1e-6.
    node_order : array-like | None, optional
        Node order for consistent indexing. If None, uses graph node order. Default is None.
    max_workers : int | None, optional
        Maximum threads when parallelizing single-node processing. Defaults to ``None`` for executor default.
    """

    if not np.any(np.asarray(node_partition) != -1):
      raise ValueError("node_partition must have -1 for single nodes.")
    
    self.G = G

    if labels is not None:
      if len(labels) != G.number_of_nodes():
        raise ValueError("labels length must equal number of graph nodes")
      self.labels = np.asarray(labels)

    self.N = G.number_of_nodes()
    self.node_partition = np.asarray(node_partition)
    self.undirected = undirected
    self.similarity_index = similarity_index
    self.tie_policy = tie_policy
    self.eps = float(eps)
    self.node_order = np.asarray(node_order) if node_order is not None else None
    self.max_workers = max_workers

    # Validate accepted arguments up front
    accepted_indices = {
      "hellinger_similarity",
      "cosine_similarity",
      "pearson_correlation",
      "weighted_jaccard",
      "jaccard_probability",
      "tanimoto_coefficient",
    }
    if self.similarity_index not in accepted_indices:
      raise ValueError(f"similarity_index must be one of {sorted(accepted_indices)}")

    accepted_policies = {
      "cluster_only",
      "include_equal",
      "include_equal_same_cluster",
      "deterministic",
    }
    if self.tie_policy not in accepted_policies:
      raise ValueError(f"tie_policy must be one of {sorted(accepted_policies)}")

  def fit(self, source_sim_matrix : npt.NDArray, target_sim_matrix : npt.NDArray):
    """
    Assign covers to single nodes and compute NOCs.

    Steps
    -----
      1) Transform similarities to distances according to `index`.
      2) Collect neighbor set of the single node in the chosen direction.
      3) For each non-trivial community, compute average distance to its neighbors;
      use max distance if there are no neighbors in that community.
      4) Select candidate covers where distance < max_dist (finite).
      5) If >1 candidate, build COST=|Δ distance| matrix, cluster with complete
          linkage, cut at max jump, and select covers tied to the closest.
      6) Compute a similarity-proxy score for each selected cover.

    Parameters
    ----------
      source_sim_matrix: NxN similarity for source/outgoing perspective.
      target_sim_matrix: NxN similarity for target/incoming perspective.

    Populates
    ----------
      - node_cover_partition:
          Hard assignments where unique; otherwise -1 for NOCs.
      - single_node_cover_map:
          Maps single nodes to their cover nodes.
      - single_nodes_cover_scores:
          Stores scores for single node covers.
    """
    from ..utils import match
    # Validate partition length and matrix shapes
    if self.node_partition.shape[0] != self.N:
      raise ValueError("node_partition length must equal number of graph nodes")
    if source_sim_matrix.shape != (self.N, self.N):
      raise ValueError("source_sim_matrix must be of shape (N, N)")
    if target_sim_matrix.shape != (self.N, self.N):
      raise ValueError("target_sim_matrix must be of shape (N, N)")

    # Build consistent node ordering and index mapping
    nodes_sorted_graph = np.array(np.sort(list(self.G.nodes())))
    if self.node_order is None:
      self.nodes_sorted = nodes_sorted_graph
    else:
      if set(self.node_order.tolist()) != set(nodes_sorted_graph.tolist()):
        raise ValueError("node_order must contain exactly the graph nodes")
      self.nodes_sorted = np.asarray(self.node_order)
    self.node_to_index = {n: i for i, n in enumerate(self.nodes_sorted)}

    self.node_cover_partition = self.node_partition.copy()

    # Edgelist with both node ids and index-mapped columns for fast lookup
    edgelist = [
      {
        "source": u,
        "target": v,
        "source_idx": self.node_to_index[u],
        "target_idx": self.node_to_index[v],
        "weight": data.get("weight", 1.0),
      }
      for u, v, data in self.G.edges(data=True)
    ]
    self.edgelist = pd.DataFrame(edgelist)

    # Compute cover candidates and scores from both directions
    nocs_src, nocs_scores_src = self.compute_nocs(source_sim_matrix, target_sim_matrix, "source", self.similarity_index)
    nocs_tgt, nocs_scores_tgt = self.compute_nocs(source_sim_matrix, target_sim_matrix, "target", self.similarity_index)

    # Previously combined cover ids were computed but unused; omit now

    self.single_node_cover_map = nocs_src.copy()

    for key, value in nocs_tgt.items():
      if key not in self.single_node_cover_map.keys():
        self.single_node_cover_map[key] = value
      else:
        self.single_node_cover_map[key] += value
        self.single_node_cover_map[key] = list(set(self.single_node_cover_map[key]))

    self.single_nodes_cover_scores = nocs_scores_src.copy()

    for key, value in nocs_scores_tgt.items():
      if key not in self.single_nodes_cover_scores.keys():
        self.single_nodes_cover_scores[key] = value
      else:
        for key2, value2 in nocs_scores_tgt[key].items():
          if key2 not in self.single_nodes_cover_scores[key].keys():
            self.single_nodes_cover_scores[key].update({key2 : value2})
          else:
            self.single_nodes_cover_scores[key][key2] = 0.5 * (value2 + self.single_nodes_cover_scores[key][key2])  # average score betwen source and target

    not_nocs = []

    for key in self.single_node_cover_map.keys():
      if len(self.single_node_cover_map[key]) == 1:
        not_nocs.append(key)
      i = match([key], self.labels)
      if len(self.single_node_cover_map[key]) == 1 and self.node_cover_partition[i] == -1:
        self.node_cover_partition[i] = self.single_node_cover_map[key][0]

    for key in not_nocs:
      del self.single_node_cover_map[key]
      del self.single_nodes_cover_scores[key]

    # Exchange numeric keys in single_node_cover_map and single_nodes_cover_scores to their respective label from self.labels
    if hasattr(self, "labels"):
      label_map = {i: self.labels[i] for i in range(len(self.labels))}
    else:
      label_map = {i: self.nodes_sorted[i] for i in range(len(self.nodes_sorted))}

    def relabel_dict_keys(d):
      return {label_map.get(k, k): v for k, v in d.items()}

    self.single_node_cover_map = relabel_dict_keys(self.single_node_cover_map)
    self.single_nodes_cover_scores = relabel_dict_keys(self.single_nodes_cover_scores)

  def compute_nocs(self, source_sim_matrix, target_sim_matrix, direction, index):
    """
    Compute per-single-node cover candidates and scores from one direction.

    Steps:
      1) Transform similarities to distances according to `index`.
      2) Collect neighbor set of the single node in the chosen direction.
      3) For each non-trivial community, compute average distance to its neighbors;
         use max distance if there are no neighbors in that community.
      4) Select candidate covers where distance < max_dist (finite).
      5) If >1 candidate, build COST=|Δ distance| matrix, cluster with complete
         linkage, cut at max jump, and select covers tied to the closest.
      6) Compute a similarity-proxy score for each selected cover.

    Returns
    -------
      single_nodes_covers: dict[node_label] -> list of cover community ids
      single_nodes_scores: dict[node_label] -> {cover_id: score}
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from ..utils import fast_cut_tree

    single_nodes_covers = {}    # dictionary of single node to scores of assigned covers
    single_nodes_scores = {}    # dictionary of single node to assigned covers

    # 1) Distance transform per accepted index family
    if index == "hellinger_similarity" or index == "cosine_similarity" or index == "pearson_correlation":
      max_dist = np.sqrt(2)
      if direction == "source":
        distance_matrix = np.sqrt(2 * (1 - source_sim_matrix))
      elif direction == "target":
        distance_matrix = np.sqrt(2 * (1 - target_sim_matrix))
      else:
        raise ValueError("No accepted direction.")
    elif index == "weighted_jaccard" or index == "jaccard_probability" or index == "tanimoto_coefficient":
      max_dist = 1.0
      if direction == "source":
        distance_matrix = 1 - source_sim_matrix
      elif direction == "target":
        distance_matrix = 1 - target_sim_matrix
      else:
        raise ValueError("No accepted direction.")
    else: raise ValueError("No accepted index.")

    ## Single nodes ----
    single_nodes = np.where(self.node_partition == -1)[0]
    ## Nodes with single community membership ----
    non_single_nodes_map = [(set(np.where(self.node_partition == i)[0]), i) for i in np.unique(self.node_partition) if i != -1]

    def process_single_node(sidx: int):
      # 2) Directional neighbor set for sidx (in index space)
      if direction == "source":
        distance_neighbors = set(self.edgelist.loc[self.edgelist["source_idx"] == sidx]["target_idx"])
      elif direction == "target":
        distance_neighbors = set(self.edgelist.loc[self.edgelist["target_idx"] == sidx]["source_idx"])
      else:
        raise ValueError("No accepted direction.")

      # 3) Average distance from sidx to each community (over neighbors in it)
      distance_sidx_to_nontrivial_communities = np.zeros((len(non_single_nodes_map)))

      for ii, non_single_nodes in enumerate(non_single_nodes_map):
        neighbor_nodes = list(distance_neighbors.intersection(non_single_nodes[0]))

        if len(neighbor_nodes) > 0:
          distance_sidx_to_nontrivial_communities[ii] = np.mean(distance_matrix[sidx, neighbor_nodes])
        else:
          distance_sidx_to_nontrivial_communities[ii] = max_dist

      # 4) Candidates: finite average distance
      is_covers = distance_sidx_to_nontrivial_communities < max_dist

      if not np.any(is_covers):
        return None

      candidate_pos = np.flatnonzero(is_covers)
      dists_cand = distance_sidx_to_nontrivial_communities[is_covers]
      number_neighbor_nontrivial_communities = dists_cand.shape[0]
      closer_pos = int(np.argmin(dists_cand))
      closer_community_distance = dists_cand[closer_pos]

      if number_neighbor_nontrivial_communities > 1:
        # 5) COST of pairwise |Δ distance| between candidate communities
        COST = np.zeros((number_neighbor_nontrivial_communities, number_neighbor_nontrivial_communities))
        for kk in np.arange(number_neighbor_nontrivial_communities):
          for ki in np.arange(kk+1, number_neighbor_nontrivial_communities):
            COST[kk, ki] = np.abs(dists_cand[kk] - dists_cand[ki])
            COST[ki, kk] = COST[kk, ki]

        # Hierarchical clustering of COST matrix
        # Complete linkage is appropriate for |Δ distance| clustering
        # since it tends to produce compact, spherical clusters.
        COST : npt.NDArray = linkage(squareform(COST), method="complete")

        dD = np.diff(COST[:, 2])
        if dD.shape[0] > 1:
          maximimum_height_step = int(np.argmax(dD)) 
        else:
          maximimum_height_step = 0
          
        # Cut the tree at that height to form flat clusters
        cover_partition = fast_cut_tree(COST, height=COST[maximimum_height_step, 2])
        cover_partition_closer_value = cover_partition[closer_pos]
      else:
        cover_partition = np.array([0])
        cover_partition_closer_value =  0

      # 6) Select candidate positions per tie_policy
      if self.tie_policy == "cluster_only":
        selected_pos = np.where(cover_partition == cover_partition_closer_value)[0]
      elif self.tie_policy == "include_equal":
        in_cluster = set(np.where(cover_partition == cover_partition_closer_value)[0].tolist())
        equals_any = set(np.where(np.abs(dists_cand - closer_community_distance) <= self.eps)[0].tolist())
        selected_pos = np.array(sorted(in_cluster.union(equals_any)))
      elif self.tie_policy == "include_equal_same_cluster":
        mask_same = (cover_partition == cover_partition_closer_value) & (np.abs(dists_cand - closer_community_distance) <= self.eps)
        selected_pos = np.where(mask_same)[0]
      elif self.tie_policy == "deterministic":
        # Single winner: min distance; tie-break by smallest community id
        min_mask = np.abs(dists_cand - np.min(dists_cand)) <= self.eps
        candidate_min_pos = np.where(min_mask)[0]
        if candidate_min_pos.size > 1:
          ids = np.array([non_single_nodes_map[candidate_pos[p]][1] for p in candidate_min_pos])
          candidate_min_pos = np.array([candidate_min_pos[np.argmin(ids)]])
        selected_pos = candidate_min_pos
      else:
        selected_pos = np.array([closer_pos])

      node_label = self.labels[sidx]
      node_covers = []
      node_scores = {}

      # Map selected positions back to overall community indices and record
      for pos in selected_pos:
        overall_idx = int(candidate_pos[pos])
        community_id = non_single_nodes_map[overall_idx][1]
        node_covers.append(community_id)
        if index == "hellinger_similarity" or index == "cosine_similarity" or index == "pearson_correlation":
          node_scores[community_id] = 1 - np.power(distance_sidx_to_nontrivial_communities[overall_idx]/max_dist, 2)
        elif index == "weighted_jaccard" or index == "jaccard_probability" or index == "tanimoto_coefficient":
          node_scores[community_id] = 1 - distance_sidx_to_nontrivial_communities[overall_idx]

      return node_label, node_covers, node_scores

    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      for result in executor.map(process_single_node, single_nodes):
        if result is None:
          continue
        node_label, node_covers, node_scores = result
        single_nodes_covers[node_label] = node_covers
        single_nodes_scores[node_label] = node_scores

    return  single_nodes_covers, single_nodes_scores
