# Standard libs ----
import numpy as np
import pandas as pd
import networkx as nx

#  Framework libs ----
from similarity import LinkSimilarity
import link_hierarchy_statistics_cpp as link_stats
import node_community_hierarchy_cpp as node_builder
import utils_cpp


class Clustering:
  def __init__(
    self, G : nx.DiGraph | nx.Graph, linkage="single", similarity_index="hellinger_similarity"
  ):
    self.G = G

    if isinstance(G, nx.DiGraph):
      self.undirected = False
    else:
      self.undirected = True

    # Get intersection of source and target nodes
    sources = set(u for u, _, _ in self.G.edges(data=True))
    targets = set(v for _, v, _ in self.G.edges(data=True))
    intersection_nodes = sources & targets

    # Create subgraph with only intersection nodes
    subgraph = self.G.subgraph(intersection_nodes)

    self.N = subgraph.number_of_nodes()  # Nodes in intersection
    self.M = subgraph.number_of_edges()  # Edges between intersection nodes

    self.linkage = linkage
    self.similarity_index = similarity_index

    self.linksim_condense_matrix = None
    self.linksim_edglist = None
    self.dist_mat = None
    self.dist_edgelist = None

    # Get edge list with source, target, and weight as a pandas DataFrame
    edgelist = [
      {"source": u, "target": v, "weight": data.get("weight", 1.0)}
      for u, v, data in self.G.edges(data=True)
    ]
    self.edgelist = pd.DataFrame(edgelist)

  def add_labels(self, labels: dict):
    """
    Add labels to nodes in self.G.

    Args:
      labels (dict): A dictionary mapping node IDs to labels.
    """
    nx.set_node_attributes(self.G, labels, name="label")

  def fit_linksim_matrix(self):
    LS = LinkSimilarity(self.edgelist, self.N, self.M, similarity_index=self.similarity_index, undirected=self.undirected)
    LS.similarity_linksim_matrix()
    self.linksim_condense_matrix = LS.linksim_condense_matrix

  def fit_linksim_edgelist(self):
    LS = LinkSimilarity(self.edgelist, self.N, self.M, similarity_index=self.similarity_index, undirected=self.undirected)
    LS.similarity_linksim_edgelist()
    self.linksim_edgelist = np.array(LS.linksim_edgelist)

  def delete_linksim_matrix(self):
    self.linksim_matrix = None
  
  def delete_linksim_edgelist(self):
    self.linksim_edglist = None

  def delete_dist_matrix(self):
    self.dist_mat = None

  def delete_dist_edgelist(self):
    self.dist_edgelist = None

  def get_hierarchy_matrix(self):
    from scipy.cluster.hierarchy import linkage
    return linkage(self.dist_mat, self.linkage)
  
  def get_hierarchy_edgelist(self, max_dist : float = 1):
    return np.array(utils_cpp.mst_edges_to_linkage(self.M, self.dist_edgelist, max_dist))

  def compute_features_matrix(self, linkage : int):
    features = link_stats.core(
      self.N,
      self.M,
      self.edgelist["source"].to_numpy()[:self.M],
      self.edgelist["target"].to_numpy()[:self.M],
      linkage,
      self.undirected
    )
    features.fit_matrix(self.dist_mat)
    result = np.array(
      [
        features.get_K(),
        features.get_Height(),
        features.get_D(),
        features.get_S(),
      ]
    )
    return result
  
  def compute_features_edgelist(self, linkage : int, max_dist=1):
    features = link_stats.core(
      self.N,
      self.M,
      self.edgelist["source"].to_numpy().astype(np.int32)[:self.M],
      self.edgelist["target"].to_numpy().astype(np.int32)[:self.M],
      linkage,
      self.undirected
    )
    features.fit_edgelist(self.dist_edgelist, max_dist)
    result = np.array(
      [
        features.get_K(),
        features.get_Height(),
        features.get_D(),
        features.get_S(),
      ]
    )
    return result
  
  def process_features_matrix(self):
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")

    features = self.compute_features_matrix(linkage)
    return pd.DataFrame(
        {
          "K" : features[0, :],
          "height" : features[1, :],
          "D" : features[2, :],
          "S" : features[3, :],
        }
      )

  def process_features_edgelist(self, max_dist=1):
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")

    features = self.compute_features_edgelist(linkage, max_dist=max_dist)
    return pd.DataFrame(
        {
          "K" : features[0, :],
          "height" : features[1, :],
          "D" : features[2, :],
          "S" : features[3, :],
        }
      )

  def node_community_hierarchy_matrix(self):
    if self.linkage == "single":
      linkage = 0
    else:
      raise ValueError("Link community model has not been tested with the input linkage.")

    if isinstance(self.undirected, bool):
      if self.undirected: undirected = 1
      else: undirected = 0

    NH = node_builder.core(
      self.N,
      self.M,
      self.edgelist["source"].to_numpy().astype(np.int32),
      self.edgelist["target"].to_numpy().astype(np.int32),
      linkage,
      undirected
    )

    NH.fit_matrix(self.dist_mat)

    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)

  def node_community_hierarchy_edgelist(self, undirected=None, max_dist=1):
    if self.linkage == "single":
      linkage = 0
    else:
      raise ValueError("Link community model has not been tested with the input linkage.")

    if isinstance(self.undirected, bool):
      if self.undirected: undirected = 1
      else: undirected = 0

    NH = node_builder.core(
      self.N,
      self.M,
      self.edgelist["source"].to_numpy().astype(np.int32),
      self.edgelist["target"].to_numpy().astype(np.int32),
      linkage,
      undirected
    )

    NH.fit_edgelist(self.dist_edgelist, max_dist)

    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)
    