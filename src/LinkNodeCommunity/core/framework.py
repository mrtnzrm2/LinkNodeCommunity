"""
src/linknode/core/framework.py

Module: linknode
Author: Jorge S. Martinez Armas

Overview:
---------
Clustering provides a lightweight framework around link-based similarity and
hierarchical community construction. It prepares an edge list from the input
graph, computes link–link similarity (via a C++ backend), derives distance
representations, and exposes helpers to obtain link and node hierarchies and
basic linkage features. For directed graphs, it restricts N and M to the
intersection of nodes that appear as both sources and targets while still using
the full network’s edges to estimate similarities.

Parameters (Clustering):
------------------------
Clustering(G, linkage="single", similarity_index="hellinger_similarity")

- G (nx.Graph | nx.DiGraph): Input graph. Edge weight is taken from the
  "weight" attribute when present, otherwise defaults to 1.0.
- linkage (str): Hierarchical linkage used downstream. Supported values are
  {single, average}; other values raise a ValueError in downstream steps.
- similarity_index (str): One of {hellinger_similarity, cosine_similarity,
  pearson_correlation, weighted_jaccard, jaccard_probability,
  tanimoto_coefficient}. Passed to LinkSimilarity to control how link
  similarities are computed before distance transforms.

Goals:
------
- Prepare an edge list DataFrame from the graph with source, target, weight.
- Compute LinkSim outputs (condensed matrix or edgelist) via C++ backends.
- Build link and node hierarchies from distance data.
- Compute summary features (K, height, D, S) for link hierarchies.

Notes:
------
- Directed handling: N and M are taken from the subgraph induced by nodes that
  are both sources and targets; similarity estimation uses the full edge list
  for better statistics. The `undirected` flag is inferred from the graph type.
- Distance data: Attributes `dist_mat` and `dist_edgelist` are expected to be
  populated (from LinkSim outputs) before calling hierarchy/feature routines.
- Linkage support: Only "single" is implemented in feature and
  node-hierarchy routines; other options raise a ValueError.
- Backends: Uses C++ modules `link_hierarchy_statistics_cpp`,
  `node_community_hierarchy_cpp`, and `utils_cpp` for performance-critical
  operations. Link similarity is provided by `LinkSimilarity`.
- Outputs: Sets `linksim_condense_matrix` or `linksim_edgelist`, and after
  node hierarchy routines, `Z` (linkage array) and `linknode_equivalence` (component map).
"""

# Standard libs ----
from platform import node
import numpy as np
import pandas as pd
import networkx as nx

#  Framework libs ----
from .similarity import LinkSimilarity
from .tonewick import LinkageToNewick
from ..utils import edgelist_from_graph

# C++ libs ----
import link_hierarchy_statistics_cpp as link_stats
import node_community_hierarchy_cpp as node_builder
import utils_cpp


class Clustering:
    """
    Clustering framework for analyzing node and edge similarities in (di)graphs.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
      Input graph. Edge weights are taken from the "weight" attribute if present, otherwise default to 1.0.
    linkage : str, optional
      Hierarchical linkage criterion to use for clustering. Supported: {"single", "average"}. Default is "single".
    similarity_index : str, optional
      Similarity index for measuring edge similarity. Supported: {"hellinger_similarity", "cosine_similarity", "pearson_correlation", "weighted_jaccard", "jaccard_probability", "tanimoto_coefficient"}. Default is "hellinger_similarity".

    Attributes
    ----------
    G : nx.Graph or nx.DiGraph
      The input graph.
    undirected : bool
      True if the graph is undirected, False if directed.
    N : int
      Number of nodes in the intersection of source and target nodes (for directed graphs).
    M : int
      Number of edges between intersection nodes (for directed graphs).
    linkage : str
      The linkage criterion for clustering.
    similarity_index : str
      The similarity index for edge similarity.
    linksim_condense_matrix : array-like or None
    linksim : LinkSimilarity or None
      Instance of LinkSimilarity used for computing link similarities.
      (set by fit_linksim_matrix() or fit_linksim_edgelist())
      Condensed matrix of link similarities (computed by fit_linksim_matrix()).
    linksim_edgelist : array-like or None
      Edge list of link similarities (computed by fit_linksim_edgelist()).
    linkdist_matrix : array-like or None
      Condensed distance matrix (computed by fit_linkdist_matrix()).
    linkdist_edgelist : array-like or None
      Edge list of link distances (computed by fit_linkdist_edgelist()).
    edgelist : pandas.DataFrame
      DataFrame containing source, target, and weight for each edge in the graph.

    Notes
    -----
    - For directed graphs, only nodes that are both sources and targets are considered for clustering (intersection nodes).
    - The full edge list from the entire graph is used to compute similarities for better statistics.
    """
    def __init__(
        self, G: nx.DiGraph | nx.Graph, linkage="single", similarity_index="hellinger_similarity", consider_subgraph=False
    ):
      self.G = G

      # Check for NaN edge weights
      for u, v, data in self.G.edges(data=True):
        weight = data.get("weight", 1.0)
        if pd.isna(weight):
          raise ValueError(f"Edge ({u}, {v}) has NaN as weight. Please clean your graph.")
        
      if isinstance(G, nx.DiGraph):
        self.undirected = False
      else:
        self.undirected = True

      if consider_subgraph:
        # Get intersection of source and target nodes
        sources = set(u for u, _, _ in self.G.edges(data=True))
        targets = set(v for _, v, _ in self.G.edges(data=True))
        intersection_nodes = sources & targets

        # Create subgraph with only intersection nodes (in case of directed graph)
        subgraph = self.G.subgraph(intersection_nodes)

        self.N = subgraph.number_of_nodes()  # Nodes in intersection
        self.M = subgraph.number_of_edges()  # Edges between intersection nodes
      else:
        self.N = self.G.number_of_nodes()
        self.M = self.G.number_of_edges()

      self.linkage = linkage
      self.similarity_index = similarity_index

      self.linksim_condense_matrix = None
      self.linksim_edgelist = None
      self.linkdist_matrix = None
      self.linkdist_edgelist = None

      # Get edge list with source, target, and weight as a pandas DataFrame
      # from the whole network to have better estimates of similarities
      self.edgelist = edgelist_from_graph(self.G, sort=True)

    def add_labels(self, labels: dict):
      """
      Add labels to nodes in self.G.

      Args:
        labels (dict): A dictionary mapping node IDs to labels.
      """
      for node, label in labels.items():
          self.G.nodes[node]["label"] = label

    def fit(self, use_parallel=False, flat_mode=False, method="matrix"):
      """
      Fit the clustering model by computing link similarities and distances.

      Parameters
      ----------
      use_parallel : bool, optional
        Whether to use parallel computation for similarity calculations. Default is False.
      flat_mode : bool, optional
        If True, maps zero feature vectors to similarity 0. Default is False.
      method : str, optional
        Method to compute similarities: "matrix" for condensed matrix, "edgelist" for edge list. Default is "matrix".

      Raises
      ------
      ValueError
        If an unsupported method is provided.
      """
      if method == "matrix":
        self.fit_linksim_matrix(use_parallel=use_parallel, flat_mode=flat_mode)
        self.fit_linkdist_matrix()
      elif method == "edgelist":
        self.fit_linksim_edgelist(use_parallel=use_parallel, flat_mode=flat_mode)
        self.fit_linkdist_edgelist()
      else:
        raise ValueError("Unsupported method. Use 'matrix' or 'edgelist'.")

    def fit_linksim_matrix(self, use_parallel=False, flat_mode=False):
      self.linksim = LinkSimilarity(
        self.edgelist, self.N, self.M, similarity_index=self.similarity_index,
        undirected=self.undirected, use_parallel=use_parallel,
        flat_mode=flat_mode
      )
      self.linksim.similarity_linksim_matrix()
      self.linksim_condense_matrix = self.linksim.linksim_condense_matrix

    def fit_linksim_edgelist(self, use_parallel=False, flat_mode=False):
      self.linksim = LinkSimilarity(
        self.edgelist, self.N, self.M, similarity_index=self.similarity_index,
        undirected=self.undirected, use_parallel=use_parallel,
        flat_mode=flat_mode
      )
      self.linksim.similarity_linksim_edgelist()
      self.linksim_edgelist = np.array(self.linksim.linksim_edgelist)
    
    def fit_linkdist_matrix(self):
      if self.linksim_condense_matrix is None:
        raise ValueError("linksim_condense_matrix is not set. Run fit_linksim_matrix() first.")
      
      self.linkdist_matrix = 1 - self.linksim_condense_matrix
    
    def fit_linkdist_edgelist(self):
      if self.linksim_edgelist is None:
        raise ValueError("linksim_edgelist is not set. Run fit_linksim_edgelist() first.")
      
      self.linkdist_edgelist = np.array(self.linksim_edgelist)
      self.linkdist_edgelist[:, 2] = 1 - self.linkdist_edgelist[:, 2]

    def delete_linksim_matrix(self):
      self.linksim_matrix = None

    def delete_linksim_edgelist(self):
      self.linksim_edglist = None

    def delete_dist_matrix(self):
      self.linkdist_matrix = None

    def delete_dist_edgelist(self):
      self.linkdist_edgelist = None

    def get_hierarchy_matrix(self):
      from scipy.cluster.hierarchy import linkage
      return linkage(self.linkdist_matrix, self.linkage)

    def get_hierarchy_edgelist(self, max_dist : float = 1):
      return np.array(utils_cpp.mst_edges_to_linkage(self.M, self.linkdist_edgelist, max_dist))

    def compute_features_matrix(self, linkage : int):
      if self.edgelist.shape[0] > 1:
        sources = self.edgelist["source"].to_numpy()
        targets = self.edgelist["target"].to_numpy()
        lex_order = np.lexsort((targets, sources))
        if not np.array_equal(lex_order, np.arange(lex_order.size)):
          raise AssertionError(
            "edgelist is not sorted by 'source' and then 'target'. Sort it before computing features."
          )


      features = link_stats.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32)[:self.M],
        self.edgelist["target"].to_numpy().astype(np.int32)[:self.M],
        linkage,
        self.undirected
      )
      features.fit_matrix(self.linkdist_matrix)
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
      if self.linkdist_edgelist is None:
        raise ValueError("linkdist_edgelist is not set. Run fit_linksim_edgelist() and build the distance column first.")

      if self.linkdist_edgelist.ndim != 2 or self.linkdist_edgelist.shape[1] != 3:
        raise ValueError("linkdist_edgelist must be a (n_edges, 3) array [edge_i, edge_j, distance].")

      if self.linksim_edgelist is not None:
        link_indices = self.linkdist_edgelist[:, :2]
        expected_indices = self.linksim_edgelist[:, :2]
        if not np.array_equal(link_indices, expected_indices):
          raise AssertionError(
            "linkdist_edgelist first two columns differ from linksim_edgelist. Did you modify edge indices when converting to distances?"
          )

      if self.edgelist.shape[0] > 1:
        sources = self.edgelist["source"].to_numpy()
        targets = self.edgelist["target"].to_numpy()
        lex_order = np.lexsort((targets, sources))
        if not np.array_equal(lex_order, np.arange(lex_order.size)):
          raise AssertionError(
            "edgelist is not sorted by 'source' and then 'target'. Sort it before computing features."
          )

      features = link_stats.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32)[:self.M],
        self.edgelist["target"].to_numpy().astype(np.int32)[:self.M],
        linkage,
        self.undirected
      )
      features.fit_edgelist(self.linkdist_edgelist, max_dist)
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
      else:
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
      else:
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

    def node_community_hierarchy_matrix(self, use_parallel=False):
      if self.linkage == "single":
        linkage = 0
      else:
        raise ValueError("Link community model has not been tested with the input linkage.")

      if self.edgelist.shape[0] > 1:
        sources = self.edgelist["source"].to_numpy()
        targets = self.edgelist["target"].to_numpy()
        lex_order = np.lexsort((targets, sources))
        if not np.array_equal(lex_order, np.arange(lex_order.size)):
          raise AssertionError(
            "edgelist is not sorted by 'source' and then 'target'. Sort it before computing features."
          )

      if isinstance(self.undirected, bool):
        if self.undirected: undirected = 1
        else: undirected = 0

      NH = node_builder.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        undirected,
        use_parallel=use_parallel
      )

      NH.fit_matrix(self.linkdist_matrix)

      self.Z = NH.get_node_hierarchy()
      self.Z = np.array(self.Z)
      self.linknode_equivalence = NH.get_linknode_equivalence()
      self.linknode_equivalence = np.array(self.linknode_equivalence)

    def node_community_hierarchy_edgelist(self, use_parallel=False, max_dist=1):
      if self.linkage == "single":
        linkage = 0
      else:
        raise ValueError("Link community model has not been tested with the input linkage.")
      
      if self.linksim_edgelist is not None:
        link_indices = self.linkdist_edgelist[:, :2]
        expected_indices = self.linksim_edgelist[:, :2]
        if not np.array_equal(link_indices, expected_indices):
          raise AssertionError(
            "linkdist_edgelist first two columns differ from linksim_edgelist. Did you modify edge indices when converting to distances?"
          )
      
      if self.edgelist.shape[0] > 1:
        sources = self.edgelist["source"].to_numpy()
        targets = self.edgelist["target"].to_numpy()
        lex_order = np.lexsort((targets, sources))
        if not np.array_equal(lex_order, np.arange(lex_order.size)):
          raise AssertionError(
            "edgelist is not sorted by 'source' and then 'target'. Sort it before computing features."
          )

      if isinstance(self.undirected, bool):
        if self.undirected: undirected = 1
        else: undirected = 0

      NH = node_builder.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        undirected,
        use_parallel=use_parallel
      )

      NH.fit_edgelist(self.linkdist_edgelist, max_dist)

      self.Z = NH.get_node_hierarchy()
      self.Z = np.array(self.Z)
      self.linknode_equivalence = NH.get_linknode_equivalence()
      self.linknode_equivalence = np.array(self.linknode_equivalence)

    def to_newick(self, labels=None, branch_length=True):
      if not hasattr(self, 'Z'):
        raise ValueError("Hierarchy not computed. Run node_community_hierarchy_matrix() or node_community_hierarchy_edgelist() first.")

      TN = LinkageToNewick(self.Z, labels=labels, branch_length=branch_length)
      TN.fit()
      self.newick = TN.newick
    
