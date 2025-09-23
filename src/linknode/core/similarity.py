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