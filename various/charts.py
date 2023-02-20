import numpy as np

class Charts:
  def __init__(self, H) -> None:
    # Get parameters ----
    self.Z = H.Z
    self.nodes = H.nodes

  # Average linkage ----

  def set_Av(self, slcc):
    self.slcc = slcc

  def average_best_chart(self, K, R, ids):
    # Define new ids ----
    nids = np.zeros(self.nodes)
    self.single_nodes_check(ids)
    from various.network_tools import sort_by_size
    _, f = sort_by_size(ids, self.nodes)
    for i, key in enumerate(f):
      w = np.where(ids == key)[0]
      nids[w] = i
    return nids.astype(int)


  # Single linkage ----
    
  def single_nodes_check(self, ids):
    from collections import Counter
    id_counter =  Counter(ids)
    for key in id_counter:
      if id_counter[key] == 1:
        ids[ids == key] = -1

  def hierarchical_chart(self, K):
    from scipy.cluster.hierarchy import cut_tree
    # Get node memberships from H ----
    nodes_ids = cut_tree(
      self.Z,
      n_clusters = K
    ).reshape(-1)
    # Assign all single nodes into the -1 membership ----
    self.single_nodes_check(nodes_ids)
    return nodes_ids
