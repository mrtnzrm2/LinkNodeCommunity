"""
src/linknode/core/similarity.py

Module: linknode
Author: Jorge S. Martinez Armas

Overview:
---------
LinkSimilarity wraps C++ backends to compute pairwise link similarities from
an input edge list. It supports two output modes: a condensed matrix and an
edge-list representation, and also exposes per-endpoint (source/target)
similarity matrices.

Parameters (LinkSimilarity):
----------------------------
LinkSimilarity(edgelist, N, M, similarity_index="hellinger_similarity", undirected=True)

- edgelist (pd.DataFrame): Edge list with columns [source, target, weight].
- N (int): Number of nodes in the analyzed subgraph.
- M (int): Number of edges in the analyzed subgraph.
- similarity_index (str): One of {hellinger_similarity, cosine_similarity,
  pearson_correlation, weighted_jaccard, jaccard_probability,
  tanimoto_coefficient}.
- undirected (bool): Whether to treat the graph as undirected for similarity.

Notes:
------
- Backends: Uses C++ module `linksim_cpp` via `linksim.core` for performance.
- API: Call `similarity_linksim_matrix()` or `similarity_linksim_edgelist()`
  to populate attributes.
- Outputs: Sets `linksim_condense_matrix` or `linksim_edgelist`, and always
  `source_sim_matrix` and `target_sim_matrix`.
"""

import numpy as np
import pandas as pd
import networkx as nx

# cpp libs ----
import linksim_cpp as linksim

class LinkSimilarity:
  def __init__(
    self, edgelist : pd.DataFrame, N, M, similarity_index="hellinger_similarity", undirected=True
  ):
    # Parameters ----
    self.edgelist = edgelist
    self.similarity_index = similarity_index
    self.N = N # Nodes of the subgraph analyzed
    self.M = M # Edges of the subgraph analyzed
    self.undirected = undirected

    self.similarity_indices_map = {
      "tanimoto_coefficient" : 0,
      "cosine_similarity" : 1,
      "jaccard_probability" : 2,
      "hellinger_similarity" : 3,
      "pearson_correlation" : 4,
      "weighted_jaccard" : 5
    }
  
  def similarity_linksim_matrix(self):
    ls = linksim.core(
      self.edgelist.to_numpy().reshape(-1, 3),
      self.N,
      self.M,
      self.similarity_indices_map[self.similarity_index]
    )

    ls.fit_linksim_condense_matrix()
    self.linksim_condense_matrix = np.array(ls.get_linksim_condense_matrix())
    self.source_sim_matrix = np.array(ls.get_source_matrix())
    self.target_sim_matrix = np.array(ls.get_target_matrix())

  def similarity_linksim_edgelist(self):
    ls = linksim.core(
      self.edgelist.to_numpy().reshape(-1, 3),
      self.N,
      self.M,
      self.similarity_indices_map[self.similarity_index]
    )

    ls.fit_linksim_edgelist()
    self.linksim_edgelist = np.array(ls.get_linksim_edgelist())
    self.source_sim_matrix = np.array(ls.get_source_matrix())
    self.target_sim_matrix = np.array(ls.get_target_matrix())
