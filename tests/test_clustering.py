import numpy as np
import networkx as nx
import pytest
from LinkNodeCommunity import Clustering, NOCFinder, LinkageToNewick, ACCEPTED_SIMILARITY_INDICES

from LinkNodeCommunity.utils import (
    is_valid_linkage_matrix,
    collapsed_partition
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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 2, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 5, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."
    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."

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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 2, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 5, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."

    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."

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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 2, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 5, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."

    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.33333333333333337]), "Height at the maximum score is incorrect."

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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 2, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 5, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.1339745962155614]), "Height at the maximum score is incorrect."

    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.33333333333333337]), "Height at the maximum score is incorrect."

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
    clustering = Clustering(G, edge_complete=False)

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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [1.0]), "Height at the maximum score is incorrect."

    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 5, "Partition length is incorrect."

    assert clustering.number_link_communities == 1, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 1, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [1.0]), "Height at the maximum score is incorrect."

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
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 6, "Number of link clusters is incorrect."

    assert clustering.number_link_communities == 3, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 4, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.2928932188134524]), "Height at the maximum score is incorrect."

    # check partition
    partition_dict = clustering.equivalence_partition(score="D")
    assert len(partition_dict) == 6, "Number of link clusters is incorrect."

    assert clustering.number_link_communities == 2, "Number of link clusters is incorrect."
    assert clustering.number_node_communities == 3, "Number of node clusters is incorrect."
    assert np.allclose(clustering.height_at_maximum, [0.2928932188134524]), "Height at the maximum score is incorrect."

    partition = collapsed_partition(list(partition_dict.values()))

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

def test_add_labels():
    """Test the add_labels method of the Clustering class."""
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

    # Add labels to the nodes
    labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}   
    clustering = Clustering(G, labels=labels)

    # Fit the clustering model
    clustering.fit(method="matrix")

    # Check if the labels have been added correctly
    assert clustering.G.nodes[0]["label"] == "A", "Label for node 0 is incorrect."
    assert clustering.G.nodes[1]["label"] == "B", "Label for node 1 is incorrect."
    assert clustering.G.nodes[2]["label"] == "C", "Label for node 2 is incorrect."
    assert clustering.G.nodes[3]["label"] == "D", "Label for node 3 is incorrect."
    assert clustering.G.nodes[4]["label"] == "E", "Label for node 4 is incorrect."

    print("Add labels test passed successfully.")


def test_add_labels_exception():
    """Test the add_labels method of the Clustering class."""
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

    try:
        # Add labels to the nodes
        labels = {0: "A", 1: "B", 2: "C", 3: "D"}  # Missing label for node 4
        _ = Clustering(G, labels=labels)
    except ValueError as e:
        assert str(e) == f"Number of labels ({len(labels)}) does not match number of nodes in the graph ({len(G.nodes)}).", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for missing labels."

    try:
        # Add labels to the nodes
        labels = {0: "A", 1: "A", 2: "C", 3: "D", 4 : "E"}  # Repeated label for node 0
        _ = Clustering(G, labels=labels)
    except ValueError as e:
        assert str(e) == "All labels must be unique.", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for missing labels."

    try:
        # Add labels to the nodes
        labels = {0: "A", 5: "B", 2: "C", 3: "D", 4 : "E"}  # Node 5 does not exist
        _ = Clustering(G, labels=labels)
    except ValueError as e:
        assert str(e) == f"Labels keys must exactly match the set of nodes in the graph.", "Incorrect exception message."
    else:
        assert False, "ValueError not raised for missing labels."

    print("Add labels exception test passed successfully.")

def test_edge_complete_subgraph_1():
    """
    Test the edge_complete_subgraph attribute of the Clustering class.
    In this test, the digraph has more source nodes than target nodes.
    """
    # Create a small directed graph with 6 nodes but 3 in the edge-complete graph
    # and 11 edges
    edge_list = np.array([
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
        [4, 0],
        [4, 2],
        [5, 1],
        [5, 2]
    ])

    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # Initialize the Clustering class
    clustering = Clustering(G, edge_complete=True)

    # Fit the clustering model
    clustering.fit(method="matrix")

    # Check if the edge_complete_subgraph attribute is correctly set
    assert clustering.edge_complete_subgraph.number_of_nodes() == clustering.N, "Number of nodes in edge_complete_subgraph is incorrect."
    assert clustering.edge_complete_subgraph.number_of_edges() == clustering.M, "Number of edges in edge_complete_subgraph is incorrect."

    # Check partition
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 3, "Partition length is incorrect."

    print("edge_complete_subgraph test passed successfully.")


def test_edge_complete_subgraph_2():
    """
    Test the edge_complete_subgraph attribute of the Clustering class.
    In this test, the digraph has more target nodes than source nodes.
    """

    # Create a small directed graph with 6 nodes but 3 in the edge-complete graph
    # and 8 edges
    edge_list = np.array([
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 0],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 5]
    ])

    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # Initialize the Clustering class
    clustering = Clustering(G, edge_complete=True)

    # Fit the clustering model
    clustering.fit(method="matrix")

    # Check if the edge_complete_subgraph attribute is correctly set
    assert clustering.edge_complete_subgraph.number_of_nodes() == clustering.N, "Number of nodes in edge_complete_subgraph is incorrect."
    assert clustering.edge_complete_subgraph.number_of_edges() == clustering.M, "Number of edges in edge_complete_subgraph is incorrect."

    # Check partition
    partition_dict = clustering.equivalence_partition(score="S")
    assert len(partition_dict) == 3, "Partition length is incorrect."

    print("edge_complete_subgraph test passed successfully.")

def test_clustering_inputs():
    """Test the input validation of the Clustering class."""
    # Test with a non-networkx graph
    try:
        _ = Clustering("not_a_graph")
    except TypeError as e:
        assert str(e) == "G must be an instance of nx.Graph or nx.DiGraph.", "Incorrect exception message."
    else:
        assert False, "Expected ValueError for non-graph input."

    # Test wrong method
    adjacency_matrix = np.array(
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]
    )
    G = nx.Graph(adjacency_matrix)
    clustering = Clustering(G)
    try:
        clustering.fit(method="invalid_method")
    except ValueError as e:
        assert str(e) == "Unsupported method. Use 'matrix' or 'edgelist'.", "Incorrect exception message."
    else:
        assert False, "Expected ValueError for invalid method."

    # Teste wrong edge_complete type
    try:
        _ = Clustering(G, edge_complete="not_a_boolean")
    except TypeError as e:
        assert str(e) == "edge_complete must be a boolean value (True or False).", "Incorrect exception message."
    else:
        assert False, "Expected TypeError for edge_complete of incorrect type."

    # Test with labels of incorrect length
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    labels = ['A', 'B']  # Incorrect length
    try:
        _ = Clustering(G, labels=labels)
    except TypeError as e:
        assert str(e) == "Labels must be provided as a dictionary mapping node IDs to labels.", "Incorrect exception message."
    else:
        assert False, "Expected TypeError for labels of incorrect type."

    # Test with string node IDs and provided labels
    G = nx.Graph()
    G.add_edges_from([('a', 'b'), ('b', 'c')])
    labels = {'a': 'A', 'b': 'B', 'c': 'C'}
    try:
        _ = Clustering(G, labels=labels)
    except ValueError as e:
        assert str(e) == "If labels are provided, node IDs in the graph must not be strings. Use integer node IDs when supplying labels.", "Incorrect exception message."
    else:
        assert False, "Expected ValueError for string node IDs with provided labels."
    
    # Test to check nans in adjacency matrix
    adjacency_matrix = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    , dtype=float)
    adjacency_matrix[0, 1] = np.nan  # Introduce a NaN
    G = nx.DiGraph(adjacency_matrix)
    try:
        _ = Clustering(G)
    except ValueError as e:
        assert str(e) == "Edge (0, 1) has NaN as weight. Please clean your graph.", "Incorrect exception message."
    else:
        assert False, "Expected ValueError for adjacency matrix with NaN values."

    # Test self-loops in the graph
    adjacency_matrix = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    )
    G = nx.DiGraph(adjacency_matrix)
    try:
        _ = Clustering(G)
    except ValueError as e:
        assert str(e) == "Input graph contains self-loops. Please remove self-loops before proceeding.", "Incorrect exception message."
    else:
        assert False, "Expected ValueError for graph with self-loops."

def test_graph_strings():
    """Test handling of string node IDs in the graph."""
    # Create a simple graph with string node IDs
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"), 
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
            ("D", "E"),
            ("E", "A")
        ]   
    )
    
    clustering = Clustering(G)

    assert [node for node in clustering.G.nodes()] == [0, 1, 2, 3, 4], "Node IDs do not match relabeled node IDs."
    assert [node["label"] for node in clustering.G.nodes.values()] == ["A", "B", "C", "D", "E"], "Node labels do not match original string IDs."

    clustering.fit(method="matrix", flat_mode=True)
    partition_dict = clustering.equivalence_partition()
    assert list(partition_dict.keys()) == ["A", "B", "C", "D", "E"], "Partition keys do not match original string IDs."


def test_directed_edge_complete_false_branch():
    """Directed graphs with edge_complete=False should keep the full graph stats."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

    clustering = Clustering(G, edge_complete=False)

    assert clustering.edge_complete is False, "edge_complete flag should remain False."
    assert clustering.N == G.number_of_nodes(), "N should count all nodes in the graph."
    assert clustering.M == G.number_of_edges(), "M should count all edges in the graph."
    assert getattr(clustering, "edge_complete_subgraph", None) is None, "edge_complete_subgraph should not be set."

    clustering.fit(method="matrix", flat_mode=True)

    partition = clustering.equivalence_partition(score="S")
    assert set(partition.keys()) == set(G.nodes()), "Partition should cover every node in the original graph."


def test_equivalence_partition_preconditions_and_invalid_score():
    """equivalence_partition should guard missing data and invalid scores."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    clustering = Clustering(G)

    with pytest.raises(ValueError, match="Link-node equivalence not computed"):
        clustering.equivalence_partition()

    clustering.linknode_equivalence = np.empty((0, 0))
    with pytest.raises(ValueError, match="Linkage statistics not computed"):
        clustering.equivalence_partition()

    clustering.fit(method="matrix")
    with pytest.raises(ValueError, match="Score must be one of"):
        clustering.equivalence_partition(score="invalid")

    print("equivalence_partition preconditions and invalid score test passed successfully.")


def test_fit_linkdist_preconditions_and_edgelist_checks():
    """Distance builders require similarity data and a sorted edgelist."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

    clustering = Clustering(G)

    with pytest.raises(ValueError, match="linksim_condense_matrix is not set"):
        clustering.fit_linkdist_matrix()

    with pytest.raises(ValueError, match="linksim_edgelist is not set"):
        clustering.fit_linkdist_edgelist()

    clustering.fit_linksim_matrix(flat_mode=True)
    clustering.fit_linkdist_matrix()
    clustering.edgelist = clustering.edgelist.iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError, match="edgelist is not sorted by 'source' and then 'target'"):
        clustering.process_features_matrix()

    clustering_edgelist = Clustering(G)
    clustering_edgelist.fit_linksim_edgelist(flat_mode=True)
    clustering_edgelist.fit_linkdist_edgelist()
    clustering_edgelist.linkdist_edgelist[[0, 1]] = clustering_edgelist.linkdist_edgelist[[1, 0]]
    with pytest.raises(AssertionError, match="linkdist_edgelist first two columns differ from linksim_edgelist"):
        clustering_edgelist.node_community_hierarchy_edgelist()
    
    print("fit_linkdist preconditions and edgelist checks test passed successfully.")


def test_clustering_to_newick_requires_hierarchy():
    """to_newick should fail without hierarchy and succeed after fit."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

    clustering = Clustering(G)
    with pytest.raises(ValueError, match="Hierarchy not computed"):
        clustering.to_newick()

    clustering.fit(method="matrix")
    clustering.to_newick()

    assert hasattr(clustering, "newick"), "Newick string attribute not created."
    assert clustering.newick.endswith(";"), "Newick string should terminate with ';'."
    
    print("clustering to_newick requires hierarchy test passed successfully.")


def test_tree_network_exception():
    """Test tree_network function."""
    G = nx.DiGraph()
    edges = [(1, 2), (2, 3), (3, 4)]
    G.add_edges_from(edges)

    try:
        _ = Clustering(G)
    except ValueError as e:
        assert str(e) == "Graph must have more than 2 nodes and more than 1 edge for clustering."
    else:
        assert False, "Expected ValueError was not raised."

    print("All assertions passed for test_tree_network.")

def test_tree_network():
    """Test tree_network function."""
    G = nx.Graph()
    edges = [(1, 2), (2, 3), (3, 4)]
    G.add_edges_from(edges)

    clustering = Clustering(G)

    try:
        clustering.fit(method='matrix')
    except Exception as e:
        assert str(e) == "Denominator zero in Sc; returning 0.", f"Unexpected exception message: {str(e)}"
    else:
        assert False, "Expected an exception but none was raised."

    print("Tree network test passed")

def test_directed_tree_network():
    """Test tree_network function."""
    G = nx.Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    G.add_edges_from(edges)

    clustering = Clustering(G)

    try:
        clustering.fit(method='matrix')
    except Exception as e:
        assert str(e) == "Denominator zero in Sc; returning 0.", f"Unexpected exception message: {str(e)}"
    else:
        assert False, "Expected an exception but none was raised."

    print("Directed tree network test passed")