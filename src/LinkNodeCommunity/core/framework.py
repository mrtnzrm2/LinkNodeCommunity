"""
Path: src/LinkNodeCommunity/core/framework.py

Module: LinkNodeCommunity.core.framework
Author: Jorge S. Martinez Armas

Overview:
---------
Coordinates the LinkNodeCommunity clustering workflow. Builds graph edgelists,
executes link similarity routines, and exposes helpers to derive link and node
hierarchies alongside diagnostic statistics.

Key Components:
---------------
- Clustering: orchestrates similarity computation, distance conversion, and
  hierarchy summaries.

Notes:
------
- Directed graphs can restrict statistics to the edge-complete subgraph via
  `consider_subgraph=True`.
- Relies on the `link_hierarchy_statistics_cpp`, `node_community_hierarchy_cpp`,
  and `utils_cpp` extensions for performance-critical operations.
"""

# Standard libs ----
import numpy as np
import pandas as pd
import networkx as nx

#  Framework libs ----
from .similarity import LinkSimilarity, ACCEPTED_SIMILARITY_INDICES
from .tonewick import LinkageToNewick
from ..utils import edgelist_from_graph, linknode_equivalence_partition

# C++ libs ----
import link_hierarchy_statistics_cpp as linkstats
import node_community_hierarchy_cpp as link2node
import utils_cpp


class Clustering:
    """
    Clustering framework for analysing node and edge similarities in (di)graphs.
    
    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        Input graph. Edge weights default to 1.0 when absent.
    linkage : str, optional
        Hierarchical linkage criterion to use ({"single", "average"}). Default is "single".
    similarity_index : str, optional
        Similarity index used by `LinkSimilarity`. Supported values include {
              "bhattacharyya_coefficient", "cosine_similarity", "pearson_correlation",
              "weighted_jaccard", "jaccard_probability", "tanimoto_coefficient"}.
                  Default is "bhattacharyya_coefficient".
    consider_subgraph : bool, optional
        When True, restricts statistics (N, M) to nodes that appear as both sources and targets while computing similarities on the full graph. Default is False.
    
    Attributes
    ----------
    G : nx.Graph | nx.DiGraph
        The input graph.
    undirected : bool
        True when the graph is undirected.
    N, M : int
        Node and edge counts used in the similarity pipeline (respecting `consider_subgraph`).
    linkage : str
        The linkage criterion passed at construction time.
    similarity_index : str
        The configured similarity index.
    edgelist : pandas.DataFrame
        Sorted edge list with columns [source, target, weight].
    linksim : LinkSimilarity | None
        Populated after calling `fit_linksim_matrix` or `fit_linksim_edgelist`.
    linksim_condense_matrix : np.ndarray | None
        Condensed link-link similarity matrix when computed.
    linksim_edgelist : pandas.DataFrame | None
        Link similarity edge list when computed.
    linkdist_matrix : np.ndarray | None
        Condensed link-link distance matrix when computed.
    linkdist_edgelist : pandas.DataFrame | None
        Link distance edge list when computed.
    
    """
    def __init__(
        self, G: nx.DiGraph | nx.Graph, similarity_index="bhattacharyya_coefficient", consider_subgraph=False
    ):
      
      if similarity_index not in ACCEPTED_SIMILARITY_INDICES:
        raise ValueError(
          f"Similarity index '{similarity_index}' is not supported.\nAccepted indices are: {ACCEPTED_SIMILARITY_INDICES}"
        )

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

    def fit(
        self,
        use_parallel=False, 
        flat_mode=False, 
        method="matrix", 
        max_dist=1.0, 
        verbose=0,
        force=False
      ):
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
        self.fit_linksim_matrix(use_parallel=use_parallel, flat_mode=flat_mode, verbose=verbose)
        self.fit_linkdist_matrix()
        self.linkstats = self.process_features_matrix(verbose=verbose, force=force)
        self.node_community_hierarchy_matrix(use_parallel=use_parallel, verbose=verbose)
      elif method == "edgelist":
        self.fit_linksim_edgelist(use_parallel=use_parallel, flat_mode=flat_mode, verbose=verbose)
        self.fit_linkdist_edgelist()
        self.linkstats = self.process_features_edgelist(max_dist=max_dist, verbose=verbose, force=force)
        self.node_community_hierarchy_edgelist(use_parallel=use_parallel, max_dist=max_dist, verbose=verbose)
      else:
        raise ValueError("Unsupported method. Use 'matrix' or 'edgelist'.")

    def fit_linksim_matrix(self, use_parallel=False, flat_mode=False, verbose=0):
      self.linksim = LinkSimilarity(
        self.edgelist, self.N, self.M, similarity_index=self.similarity_index,
        undirected=self.undirected, use_parallel=use_parallel,
        flat_mode=flat_mode
      )
      self.linksim.similarity_linksim_matrix(verbose=verbose)
      self.linksim_condense_matrix = self.linksim.linksim_condense_matrix

    def fit_linksim_edgelist(self, use_parallel=False, flat_mode=False, verbose=0):
      self.linksim = LinkSimilarity(
        self.edgelist, self.N, self.M, similarity_index=self.similarity_index,
        undirected=self.undirected, use_parallel=use_parallel,
        flat_mode=flat_mode
      )
      self.linksim.similarity_linksim_edgelist(verbose=verbose)
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
      return linkage(self.linkdist_matrix, method="single")

    def get_hierarchy_edgelist(self, max_dist : float = 1):
      return np.array(utils_cpp.mst_edges_to_linkage(self.M, self.linkdist_edgelist, max_dist))

    def compute_features_matrix(self, linkage : int, verbose=0, force=False):
      if self.edgelist.shape[0] > 1:
        sources = self.edgelist["source"].to_numpy()
        targets = self.edgelist["target"].to_numpy()
        lex_order = np.lexsort((targets, sources))
        if not np.array_equal(lex_order, np.arange(lex_order.size)):
          raise AssertionError(
            "edgelist is not sorted by 'source' and then 'target'. Sort it before computing features."
          )

      features = linkstats.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32)[:self.M],
        self.edgelist["target"].to_numpy().astype(np.int32)[:self.M],
        linkage,
        self.undirected,
        verbose=verbose,
        force=force
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

    def compute_features_edgelist(self, linkage : int, max_dist=1, verbose=0, force=False):
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

      features = linkstats.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32)[:self.M],
        self.edgelist["target"].to_numpy().astype(np.int32)[:self.M],
        linkage,
        self.undirected,
        verbose=verbose,
        force=force
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

    def process_features_matrix(self, verbose=0, force=False):
      linkage = 0  # Single linkage
                   # Only single linkage supported deliberately
      features = self.compute_features_matrix(linkage, verbose=verbose, force=force)
      return pd.DataFrame(
          {
            "K" : features[0, :],
            "height" : features[1, :],
            "D" : features[2, :],
            "S" : features[3, :],
          }
        )

    def process_features_edgelist(self, max_dist=1, verbose=0, force=False):
      linkage = 0  # Single linkage
                   # Only single linkage supported deliberately
      features = self.compute_features_edgelist(linkage, max_dist=max_dist, verbose=verbose, force=force)
      return pd.DataFrame(
          {
            "K" : features[0, :],
            "height" : features[1, :],
            "D" : features[2, :],
            "S" : features[3, :],
          }
        )

    def node_community_hierarchy_matrix(self, use_parallel=False, verbose=0):
      linkage = 0  # Single linkage
                   # Only single linkage supported deliberately
      if self.linkdist_matrix is None:
        raise ValueError("linkdist_matrix is not set. Run fit_linksim_matrix() and build the distance column first.")
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

      NH = link2node.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        undirected,
        use_parallel=use_parallel,
        verbose=verbose
      )

      NH.fit_matrix(self.linkdist_matrix)

      self.Z = NH.get_node_hierarchy()
      self.Z = np.array(self.Z)
      self.linknode_equivalence = NH.get_linknode_equivalence()
      self.linknode_equivalence = np.array(self.linknode_equivalence)

    def node_community_hierarchy_edgelist(self, use_parallel=False, max_dist=1, verbose=0):
      linkage = 0  # Single linkage
                   # Only single linkage supported deliberately
      if self.linkdist_edgelist is None:
        raise ValueError("linkdist_edgelist is not set. Run fit_linksim_edgelist() and build the distance column first.")
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

      NH = link2node.core(
        self.N,
        self.M,
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        undirected,
        use_parallel=use_parallel,
        verbose=verbose
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
    
    def equivalence_partition(self, score : str = "S"):
      """
      Derive node partitions from the link-node equivalence using a specified score.

      Parameters
      ----------
      score : str
        The score to use for partitioning. Supported values are "D" (dissimilarity)
        and "S" (similarity). Default is "S".

      Returns
      -------
      tuple
          - number_link_communities (int or np.ndarray): Number of link communities at maximum score.
          - number_node_communities (int or np.ndarray): Corresponding number of node communities.
          - height_at_maximum (float or np.ndarray): Height at which the maximum score occurs.
      """
      if not hasattr(self, 'linknode_equivalence'):
        raise ValueError("Link-node equivalence not computed. Run node_community_hierarchy_matrix() or node_community_hierarchy_edgelist() first.")
      if not hasattr(self, 'linkstats'):
        raise ValueError("Linkage statistics not computed. Run process_features_matrix() or process_features_edgelist() first.")
      if score not in {"D", "S"}:
        raise ValueError("Score must be one of {'D', 'S'}.")
     
      return linknode_equivalence_partition(score, self.linkstats, self.linknode_equivalence)
    
