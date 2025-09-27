"""
Path: src/LinkNodeCommunity/core/similarity.py

Module: LinkNodeCommunity.core.similarity
Author: Jorge S. Martinez Armas

Overview:
---------
Wrapper around the C++ link similarity engines. Produces condensed matrices or
edge-list outputs and the per-endpoint similarity matrices required downstream.

Key Components:
---------------
- LinkSimilarity: manages similarity configuration, execution, and cached
  outputs.

Notes:
------
- Supports similarity indices {bhattacharyya_coefficient, cosine_similarity,
  pearson_correlation, weighted_jaccard, jaccard_probability,
  tanimoto_coefficient}.
- Optional parallel execution (`use_parallel`) and zero-vector handling
  (`flat_mode`) control numerical edge cases.
"""

import numpy as np
import pandas as pd
import networkx as nx

# cpp libs ----
import linksim_cpp as linksim

ACCEPTED_SIMILARITY_INDICES = [
    "tanimoto_coefficient",
    "cosine_similarity",
    "jaccard_probability",
    "bhattacharyya_coefficient",
    "pearson_correlation",
    "weighted_jaccard"
]

class LinkSimilarity:
  def __init__(
    self,  N, M, edgelist : pd.DataFrame, similarity_index="bhattacharyya_coefficient", undirected=True, use_parallel=False, flat_mode=False
  ):
    # Parameters ----
    self.edgelist = edgelist
    self.similarity_index = similarity_index
    self.N = N # Nodes of the subgraph analyzed
    self.M = M # Edges of the subgraph analyzed
    self.undirected = undirected
    self.use_parallel = use_parallel
    self.flat_mode = flat_mode  # Map zero feature vectors to similarity 0 if True

    self.similarity_indices_map = {
      "tanimoto_coefficient" : 0,
      "cosine_similarity" : 1,
      "jaccard_probability" : 2,
      "bhattacharyya_coefficient" : 3,
      "pearson_correlation" : 4,
      "weighted_jaccard" : 5
    }
  
  def similarity_linksim_matrix(self, verbose=0):
    ls = linksim.core(
      self.N,
      self.M,
      self.edgelist.to_numpy().reshape(-1, 3),
      self.similarity_indices_map[self.similarity_index],
      self.undirected,
      self.use_parallel,
      self.flat_mode,
      verbose
    )

    ls.fit_linksim_condense_matrix()
    self.linksim_condense_matrix = np.array(ls.get_linksim_condense_matrix())
    self.source_sim_matrix = np.array(ls.get_source_matrix())
    self.target_sim_matrix = np.array(ls.get_target_matrix())

  def similarity_linksim_edgelist(self, verbose=0):
    ls = linksim.core(
      self.N,
      self.M,
      self.edgelist.to_numpy().reshape(-1, 3),
      self.similarity_indices_map[self.similarity_index],
      self.undirected,
      self.use_parallel,
      self.flat_mode,
      verbose
    )

    ls.fit_linksim_edgelist()
    self.linksim_edgelist = np.array(ls.get_linksim_edgelist())
    self.source_sim_matrix = np.array(ls.get_source_matrix())
    self.target_sim_matrix = np.array(ls.get_target_matrix())
