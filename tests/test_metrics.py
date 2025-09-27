import numpy as np
import utils_cpp

def test_graph_simlarity_cpp():
    """Test of similarity C++ backend."""
    u = np.array([1, 1, 0, 1, 0, 1])
    v = np.array([0, 0, 1, 0, 1, 0])

    score = utils_cpp.tanimoto_coefficient_graph(u, v, 0, 1)
    assert score == 0.0, "Tanimoto coefficient is incorrect."
    score = utils_cpp.cosine_similarity_graph(u, v, 0, 1)
    assert score == 0.0, "Cosine similarity is incorrect."
    score = utils_cpp.jaccard_probability_graph(u, v, 0, 1)
    assert score == 0.0, "Jaccard probability is incorrect."
    score = utils_cpp.bhattacharyya_coefficient_graph(u, v, 0, 1)
    assert score == 0.0, "Bhattacharyya coefficient is incorrect."
    score = utils_cpp.pearson_correlation_graph(u, v, 0, 1)
    assert score == -1.0, "Pearson correlation is incorrect."
    score = utils_cpp.weighted_jaccard_graph(u, v, 0, 1)
    assert score == 0.0, "Weighted Jaccard is incorrect."

    u = np.array([0, 1, 0, 0, 0, 0])
    v = np.array([1, 0, 0, 0, 0, 0])

    score = utils_cpp.tanimoto_coefficient_graph(u, v, 0, 1)
    assert score == 1.0, "Tanimoto coefficient is incorrect."
    score = utils_cpp.cosine_similarity_graph(u, v, 0, 1)
    assert score == 1.0, "Cosine similarity is incorrect."
    score = utils_cpp.jaccard_probability_graph(u, v, 0, 1)
    assert score == 1.0, "Jaccard probability is incorrect."
    score = utils_cpp.bhattacharyya_coefficient_graph(u, v, 0, 1)
    assert score == 1.0, "Bhattacharyya coefficient is incorrect."
    score = utils_cpp.pearson_correlation_graph(u, v, 0, 1)
    assert score == 1.0, "Pearson correlation is incorrect."
    score = utils_cpp.weighted_jaccard_graph(u, v, 0, 1)
    assert score == 1.0, "Weighted Jaccard is incorrect."

    print("Graph similarity C++ backend test passed successfully.")