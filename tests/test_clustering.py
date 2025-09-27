import numpy as np
import networkx as nx
from LinkNodeCommunity import Clustering, NOCFinder, LinkageToNewick, ACCEPTED_SIMILARITY_INDICES

from LinkNodeCommunity.utils import (
    is_valid_linkage_matrix,
    collapsed_partition,
    fast_cut_tree
)

def test_clustering_matrix_undirected():
    """Test the Clustering class with a simple example."""
    # Create a small undirected graph with 5 nodes and 8 edges
    # Adjacency matrix for a undirected graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.Graph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    # Fit the clustering model
    clustering.fit(method="matrix")
    assert is_valid_linkage_matrix(clustering.get_hierarchy_matrix()), "Hierarchy matrix is not valid."
    assert is_valid_linkage_matrix(clustering.Z), "Linkage matrix is not valid."

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 2, "Number of link clusters is incorrect."
    assert r == 5, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."

    partition = fast_cut_tree(clustering.Z, n_clusters=5)
    assert len(partition) == 5, "Partition length is incorrect."

    # Check if the clustering has been performed correctly
    assert clustering.N == 5
    assert clustering.M == 8

    print("Clustering Matrix Undirected test passed successfully.")


def test_clustering_edgelist_undirected():
    """Test the Clustering class with a simple example."""
    # Create a small directed graph with 5 nodes and 16 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.Graph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    # Fit the clustering model
    clustering.fit(method="edgelist")

    assert is_valid_linkage_matrix(clustering.get_hierarchy_edgelist()), "Hierarchy edgelist is not valid."
    assert is_valid_linkage_matrix(clustering.Z), "Linkage matrix is not valid."

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 2, "Number of link clusters is incorrect."
    assert r == 5, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."

    partition = fast_cut_tree(clustering.Z, n_clusters=5)
    assert len(partition) == 5, "Partition length is incorrect."

    # Check if the clustering has been performed correctly
    assert clustering.N == 5
    assert clustering.M == 8

    print("Clustering Edge List Undirected test passed successfully.")

def test_clustering_matrix_directed():
    """Test the Clustering class with a simple example."""
    # Create a small directed graph with 5 nodes and 16 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.DiGraph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    # Fit the clustering model
    clustering.fit(method="matrix")
    assert is_valid_linkage_matrix(clustering.get_hierarchy_matrix()), "Hierarchy matrix is not valid."
    assert is_valid_linkage_matrix(clustering.Z), "Linkage matrix is not valid."

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 2, "Number of link clusters is incorrect."
    assert r == 5, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.33333333333333337]), "Height at the maximum score is incorrect."

    partition = fast_cut_tree(clustering.Z, n_clusters=5)
    assert len(partition) == 5, "Partition length is incorrect."

    # Check if the clustering has been performed correctly
    assert clustering.N == 5
    assert clustering.M == 16

    print("Clustering Matrix Directed test passed successfully.")

def test_clustering_edgelist_directed():
    """Test the Clustering class with a simple example."""
    # Create a small directed graph with 5 nodes and 16 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.DiGraph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    # Fit the clustering model
    clustering.fit(method="edgelist")
    assert is_valid_linkage_matrix(clustering.get_hierarchy_edgelist(1)), "Hierarchy matrix is not valid."
    assert is_valid_linkage_matrix(clustering.Z), "Linkage matrix is not valid."

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 2, "Number of link clusters is incorrect."
    assert r == 5, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.1339745962155614]), "Height at the maximum score is incorrect."
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.33333333333333337]), "Height at the maximum score is incorrect."

    partition = fast_cut_tree(clustering.Z, n_clusters=5)
    assert len(partition) == 5, "Partition length is incorrect."

    # Check if the clustering has been performed correctly
    assert clustering.N == 5
    assert clustering.M == 16

    print("Clustering Edge List Directed test passed successfully.")

def test_clustering_exceptions():
    """Test the Clustering class with a simple example."""
    # Create a small directed graph with 5 nodes and 16 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.DiGraph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    try:
        clustering.fit(method="invalid_method")
    except ValueError as e:
        assert str(e) == "Unsupported method. Use 'matrix' or 'edgelist'.", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for invalid method."

    print("Clustering exceptions test passed successfully.")

def test_similarity_exceptions():
    """Test the Clustering class with a simple example."""
    # Create a small directed graph with 5 nodes and 16 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.DiGraph(adj_matrix)
    # Initialize the Clustering class
    try:
        _ = Clustering(G, similarity_index="invalid_index")
    except ValueError as e:
        assert str(e) ==  f"Similarity index 'invalid_index' is not supported.\nAccepted indices are: {ACCEPTED_SIMILARITY_INDICES}", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for invalid similarity index."

    print("Similarity exceptions test passed successfully.")

def test_flat_mode_exception():
    """Test the Clustering class with a simple example."""
    # Create a small graph with 5 nodes and 4 edges
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    # Initialize the Clustering class
    clustering = Clustering(G)

    try:
        clustering.fit(method="matrix", flat_mode=False)
    except ValueError as e:
        assert str(e) == "One or both feature vectors have all zeros, which is not allowed.", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for flat_mode=False in directed graph."
    
    print("flat_mode exception test passed successfully.")

def test_flat_mode():
    """Test the Clustering class with a simple example."""
    # Create a small graph with 5 nodes and 5 edges
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

    # Initialize the Clustering class
    clustering = Clustering(G)
    clustering.fit(method="matrix", flat_mode=True)

    assert is_valid_linkage_matrix(clustering.Z), "Linkage matrix is not valid."

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [1.0]), "Height at the maximum score is incorrect."
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 1, "Number of link clusters is incorrect."
    assert r == 1, "Number of node clusters is incorrect."
    assert np.allclose(h, [1.0]), "Height at the maximum score is incorrect."

    partition = fast_cut_tree(clustering.Z, n_clusters=5)
    assert len(partition) == 5, "Partition length is incorrect."

    print("flat_mode test passed successfully.")

def test_nocfinder():
    """Test the NOCFinder class with a simple example."""
    # Create a small directed graph with 6 nodes and 13 edges
    # Adjacency matrix for a directed graph
    adj_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0]
        ], dtype=float
    )

    G = nx.DiGraph(adj_matrix)

    # Initialize the Clustering class
    clustering = Clustering(G)

    # Fit the clustering model
    clustering.fit(method="matrix", flat_mode=True)

    # check partition
    # k -> number of link clusters
    # r -> number of node clusters
    # h -> height at the maximum score
    # check partition
    k, r, h = clustering.equivalence_partition(score="D")
    assert k == 2, "Number of link clusters is incorrect."
    assert r == 3, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.2928932188134524]), "Height at the maximum score is incorrect."

    k, r, h = clustering.equivalence_partition(score="S")
    assert k == 3, "Number of link clusters is incorrect."
    assert r == 4, "Number of node clusters is incorrect."
    assert np.allclose(h, [0.2928932188134524]), "Height at the maximum score is incorrect."
    
    partition = fast_cut_tree(clustering.Z, n_clusters=3)
    assert len(partition) == 6, "Partition length is incorrect."
    partition = collapsed_partition(partition)

    assert np.any(np.asanyarray(partition) == -1), "Partition does not contain single nodes."
    assert not np.all(np.asanyarray(partition) != -1), "Partition does not contain communities."

    # Initialize the NOCFinder class
    noc_finder = NOCFinder(G, partition)

    # Find overlapping communities
    noc_finder.fit(clustering.linksim.source_sim_matrix, clustering.linksim.target_sim_matrix)

    # Check if the overlapping communities have been found correctly
    assert len(noc_finder.single_node_cover_map.keys()) > 0, "No overlapping communities found."

    print("NOCFinder test passed successfully.")

def test_linkage_to_newick():
    """Test the LinkageToNewick class with a simple example."""
    # Create a simple linkage matrix for 5 elements
    Z = np.array([
        [0, 1, 0.1, 2],
        [2, 3, 0.2, 2],
        [4, 5, 0.3, 4],
        [6, 7, 0.4, 5]
    ])

    # Initialize the LinkageToNewick class
    linker = LinkageToNewick(Z)
    linker.fit()
    # Convert to Newick format
    newick_str = linker.newick

    # Check if the Newick string is correctly formatted
    assert newick_str.endswith(";"), "Newick string does not end with a semicolon."
    assert "(" in newick_str and ")" in newick_str, "Newick string does not contain parentheses."

    print("LinkageToNewick test passed successfully.")

def test_linkage_to_newick():
    """Test the LinkageToNewick class with a simple example."""
    # Create a simple linkage matrix for 5 elements
    Z = np.array([
        [0, 1, 0.1, 2],
        [2, 3, 0.2, 2],
        [4, 5, 0.3, 4],
        [6, 7, 0.4, 5]
    ])

    # Initialize the LinkageToNewick class
    linker = LinkageToNewick(Z)
    linker.fit()
    # Convert to Newick format
    newick_str = linker.newick

    # Check if the Newick string is correctly formatted
    assert newick_str.endswith(";"), "Newick string does not end with a semicolon."
    assert "(" in newick_str and ")" in newick_str, "Newick string does not contain parentheses."

    print("LinkageToNewick test passed successfully.")