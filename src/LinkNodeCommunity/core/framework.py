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
  `edge_complete=True`.
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
from ..utils import (
  edgelist_from_graph,
  linknode_equivalence_partition,
  cut_tree_with_validation,
  linear_partition
)

# C++ backends ----
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
    labels : dict | None, optional
        Mapping of node IDs to their labels. If None, labels are inferred from the graph.
    linkage : str, optional
        Hierarchical linkage criterion to use ({"single", "average"}). Default is "single".
    similarity_index : str, optional
        Similarity index used by `LinkSimilarity`. Supported values include {
              "bhattacharyya_coefficient", "cosine_similarity", "pearson_correlation",
              "weighted_jaccard", "jaccard_probability", "tanimoto_coefficient"}.
                  Default is "bhattacharyya_coefficient".
    edge_complete : bool, optional
        When True, restricts statistics (N, M) to nodes that appear as both sources and targets while computing similarities on the full graph. Relevant only for directed graphs. Default is True.

    Attributes
    ----------
    G : nx.Graph | nx.DiGraph
        The input graph.
    edge_complete_subgraph : nx.Graph | nx.DiGraph | None
        The subgraph induced by nodes that appear as both sources and targets when `edge_complete=True`.
    undirected : bool
        True when the graph is undirected.
    N, M : int
        Node and edge counts used in the similarity pipeline (respecting `edge_complete`).
    node_mapping : dict
        Mapping from original node IDs to their indices in the processed graph.
    inv_node_mapping : dict
        Inverse mapping from processed graph indices back to original node IDs.
    Z : np.ndarray | None
        Linkage matrix of the node hierarchy after processing.
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
    linkstats : np.ndarray | None
        Link hierarchy statistics after processing features.
    number_link_communities : int | None
        Number of link communities after node hierarchy processing.
    number_node_communities : int | None
        Number of node communities after processing.
    height_at_maximum : float | None
        Height at which the maximum link similarity occurs.
    """
    def __init__(
        self,
        G: nx.DiGraph | nx.Graph,
        labels: dict | None = None,
        similarity_index="bhattacharyya_coefficient",
        edge_complete=True
    ):
      
      # Validate inputs
      if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError("G must be an instance of nx.Graph or nx.DiGraph.")

      if labels is not None:
        if not isinstance(labels, dict):
          raise TypeError("Labels must be provided as a dictionary mapping node IDs to labels.")

      # Validate similarity index
      if similarity_index not in ACCEPTED_SIMILARITY_INDICES:
        raise ValueError(
          f"Similarity index '{similarity_index}' is not supported.\nAccepted indices are: {ACCEPTED_SIMILARITY_INDICES}"
        )
      
      if not isinstance(edge_complete, bool):
        raise TypeError("edge_complete must be a boolean value (True or False).")
      
      # Check for NaN edge weights
      for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        if pd.isna(weight):
          raise ValueError(f"Edge ({u}, {v}) has NaN as weight. Please clean your graph.")
      
      # Check for self-loops
      if any(u == v for u, v in G.edges()):
        raise ValueError("Input graph contains self-loops. Please remove self-loops before proceeding.")
      
      # If labels is not None, nodes in G must not be strings
      if labels is not None and all(isinstance(node, str) for node in G.nodes()):
        raise ValueError("If labels are provided, node IDs in the graph must not be strings. Use integer node IDs when supplying labels.")

      # Validate labels if provided
      if labels is not None:
        if len(labels) != len(G.nodes):
          raise ValueError(f"Number of labels ({len(labels)}) does not match number of nodes in the graph ({len(G.nodes)}).")

        if len(set(labels.values())) != len(labels):
          raise ValueError("All labels must be unique.")
        
        if set(G.nodes) != set(labels.keys()):
          raise ValueError("Labels keys must exactly match the set of nodes in the graph.")

      # Relabel nodes to integers if they are strings, preserving original labels
      if all(isinstance(node, str) for node in G.nodes()):
        G_copy = self.relabel_nodes_to_integers(G.copy())
        print("Node labels converted to integers (sorted by label); originals saved in each node's 'label' attribute.")
      else:
        G_copy = G.copy()
        
      # Determine if the graph is directed or undirected
      if isinstance(G, nx.DiGraph):
        self.undirected = False
      else:
        self.undirected = True
      
      # Ensure node indices are integers
      if not all(isinstance(node, int) for node in G_copy.nodes()):
        mapping = {node: idx for idx, node in enumerate(sorted(G_copy.nodes()))}
        G_copy = nx.relabel_nodes(G_copy, mapping)
      
      # Guard: if all nodes are integers, relabel them to 0..N-1 (sorted ascending)
      if all(isinstance(node, int) for node in G_copy.nodes()):
        sorted_nodes = sorted(G_copy.nodes())
        self.node_mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
        self.inv_node_mapping = {idx: node for node, idx in self.node_mapping.items()}
        G_copy = nx.relabel_nodes(G_copy, self.node_mapping)
      else:
        # Trivial node mapping and inverse mapping: map node indices from 0 to len(G_copy)-1
        self.node_mapping = {node: idx for idx, node in enumerate(sorted(G_copy.nodes()))}
        self.inv_node_mapping = {idx: node for node, idx in self.node_mapping.items()}

      self.edge_complete = edge_complete

      # If edge_complete is True, restrict N, M to nodes that appear as both sources and targets
      if self.edge_complete and not self.undirected:
        # Get intersection of source and target nodes
        sources = set(u for u, _, _ in G_copy.edges(data=True))
        targets = set(v for _, v, _ in G_copy.edges(data=True))
        intersection_nodes = sources & targets

        # Remap intersection_nodes to range [0, len(intersection_nodes)-1]
        intersection_nodes_sorted = sorted(intersection_nodes)
        intersection_mapping = {node: idx for idx, node in enumerate(intersection_nodes_sorted)}

        # Remap the rest of the nodes to range [self.N, len(G_copy)-1]
        rest_nodes = sorted(set(G_copy.nodes()) - intersection_nodes)
        rest_mapping = {node: idx + len(intersection_nodes_sorted) for idx, node in enumerate(rest_nodes)}

        # Combine mappings
        self.node_mapping = {**intersection_mapping, **rest_mapping}

        # Relabel G_copy using node_mapping
        G_remapped = nx.relabel_nodes(G_copy, self.node_mapping)
        self.G = G_remapped

        # Extract edge_complete_subgraph from range [0, len(intersection_nodes)-1]
        edge_complete_indices = list(range(len(intersection_nodes_sorted)))
        self.edge_complete_subgraph = self.G.subgraph(edge_complete_indices)

        self.N = self.edge_complete_subgraph.number_of_nodes()  # Nodes in intersection
        self.M = self.edge_complete_subgraph.number_of_edges()  # Edges between intersection nodes

      else:
        self.G = G_copy.copy()
        self.N = self.G.number_of_nodes()
        self.M = self.G.number_of_edges()

      if self.N <= 2 or self.M <= 1:
        raise ValueError("Graph must have more than 2 nodes and more than 1 edge for clustering.")
      
      # Add labels to self.G and self.edge_complete_subgraph if labels are provided
      if labels is not None:
        # Invert node_mapping: mapped_node -> original_node
        self.inv_node_mapping = {v: k for k, v in self.node_mapping.items()}

        # Build mapping: mapped_node -> label
        mapped_labels = {mapped_node: labels[original_node] for mapped_node, original_node in self.inv_node_mapping.items()}

        self.add_labels(self.G, mapped_labels)
        if self.edge_complete and not self.undirected:
          edge_complete_labels = {node: mapped_labels[node] for node in self.edge_complete_subgraph.nodes()}
          self.add_labels(self.edge_complete_subgraph, edge_complete_labels)
        
      self.similarity_index = similarity_index

      self.linksim_condense_matrix = None
      self.linksim_edgelist = None
      self.linkdist_matrix = None
      self.linkdist_edgelist = None

      # Get edge list with source, target, and weight as a pandas DataFrame
      # from the whole network to have better estimates of similarities
      self.edgelist = edgelist_from_graph(self.G, sort=True)

    def relabel_nodes_to_integers(self, G : nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
      """
      Relabels the nodes of a graph from strings to integers in sorted order,
      and assigns the original string as a 'label' attribute to each node.

      Parameters
      ----------
      G : nx.Graph or nx.DiGraph
        The input graph with string node labels.

      Returns
      -------
      nx.Graph or nx.DiGraph
        The relabeled graph with integer nodes and 'label' attributes.
      """
      sorted_labels = sorted(G.nodes())
      mapping = {label: idx for idx, label in enumerate(sorted_labels)}
      G_int = nx.relabel_nodes(G, mapping)
      for idx, label in enumerate(sorted_labels):
        G_int.nodes[idx]["label"] = label
      return G_int

    def add_labels(self, G : nx.Graph | nx.DiGraph, labels: dict):
      """
      Add labels to nodes in the given graph.

      Parameters
      ----------
        G (nx.Graph | nx.DiGraph): The input graph.
        labels (dict): A dictionary mapping node IDs to labels.
      """
      # Check if all nodes in G are present in labels
      if len(G.nodes) != len(labels):
        raise ValueError(f"Number of nodes in graph ({len(G.nodes)}) does not match number of labels ({len(labels)}).")

      # Check if all labels are unique
      if len(set(labels.values())) != len(labels):
          raise ValueError("All labels must be unique.")
      
      if set(G.nodes) != set(labels.keys()):
        raise ValueError("Labels keys must exactly match the set of nodes in the graph.")

      for node, label in labels.items():
          G.nodes[node]["label"] = label

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
        Method to compute similarities: "matrix" for condensed matrix, "edgelist" for edge list. We recommend using "edgelist" for large graphs. Default is "matrix".

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
        self.N,
        self.M,
        self.edgelist,                            # Edge List from the full graph
        similarity_index=self.similarity_index,
        undirected=self.undirected,
        use_parallel=use_parallel,
        flat_mode=flat_mode
      )
      self.linksim.similarity_linksim_matrix(verbose=verbose)
      self.linksim_condense_matrix = self.linksim.linksim_condense_matrix

    def fit_linksim_edgelist(self, use_parallel=False, flat_mode=False, verbose=0):
      self.linksim = LinkSimilarity(
        self.N,
        self.M,
        self.edgelist,                            # Edge List from the full graph
        similarity_index=self.similarity_index,
        undirected=self.undirected,
        use_parallel=use_parallel,
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
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        self.undirected,
        verbose=verbose,
        force=force,
        edge_complete=self.edge_complete
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
        self.edgelist["source"].to_numpy().astype(np.int32),
        self.edgelist["target"].to_numpy().astype(np.int32),
        linkage,
        self.undirected,
        verbose=verbose,
        force=force,
        edge_complete=self.edge_complete
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
        verbose=verbose,
        edge_complete=self.edge_complete
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
        verbose=verbose,
        edge_complete=self.edge_complete
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
      Computes the optimal node partition from the link-node equivalence using the specified score.

      Parameters
      ----------
      score : str
        The score to use for partitioning. Supported values are "D" (dissimilarity)
        and "S" (similarity). Default is "S".

      Returns
      -------
      dict
        Dictionary mapping node labels (or integer IDs) to their community membership.

      Populates
      ---------
      number_link_communities : int | npt.ArrayLike
        The optimal number of link communities determined by the partitioning.
      number_node_communities : int | npt.ArrayLike
        The optimal number of node communities determined by the partitioning.
      height_at_maximum : float | npt.ArrayLike
        The height in the dendrogram at which the optimal partition occurs.
      """

      if not hasattr(self, 'linknode_equivalence'):
        raise ValueError("Link-node equivalence not computed. Run node_community_hierarchy_matrix() or node_community_hierarchy_edgelist() first.")
      if not hasattr(self, 'linkstats'):
        raise ValueError("Linkage statistics not computed. Run process_features_matrix() or process_features_edgelist() first.")
      if score not in {"D", "S"}:
        raise ValueError("Score must be one of {'D', 'S'}.")
      
      self.number_link_communities, self.number_node_communities, \
          self.height_at_maximum = linknode_equivalence_partition(
                  score, self.linkstats, self.linknode_equivalence
                        )
     
      # Get community membership vector
      memberships = linear_partition(
        cut_tree_with_validation(
          self.Z, n_clusters=self.number_node_communities
            )
              )

      # Prepare dictionary mapping node label or integer to membership
      if self.edge_complete and not self.undirected:
        node_list = sorted(self.edge_complete_subgraph.nodes())
      else:
        node_list = sorted(self.G.nodes())

      # Map back to original node IDs if relabeling was done
      if hasattr(self, 'inv_node_mapping'):
        node_list = [self.inv_node_mapping[node] for node in node_list]

      membership_dict = {}
      for idx, node in enumerate(node_list):
          if self.edge_complete and not self.undirected:
            label = self.edge_complete_subgraph.nodes[node].get("label", node)
          else:
            label = self.G.nodes[node].get("label", node)
          membership_dict[label] = memberships[idx]

      return membership_dict

       