/*
 * cpp/linksim_cpp/src/main.cpp
 *
 * Library: linksim_cpp
 * Author: Jorge S. Martinez Armas
 *
 * Overview:
 * ----------
 * This library computes link similarities for directed, weighted graphs using a variety of
 * node-similarity measures. It builds node similarity matrices from edge features and derives
 * link similarities either as a condensed upper‑triangular vector or as an edge‑pair list.
 * Multithreading is used to accelerate the computation.
 *
 * Main Components:
 * ----------------
 * - core class: Encapsulates the computation pipeline. It extracts feature matrices from an
 *   edge list, computes node–node similarities, sorts/indexes the edges, and produces either
 *   a condensed link‑similarity matrix or a link‑to‑link edgelist.
 * - edge_struct: Lightweight struct that stores a pair of link indices and their similarity.
 * - Internal helpers (Edge, NodeNeighbors): Support efficient neighborhood traversal.
 *
 * core Class Parameters:
 * ----------------------
 * - edgelist (std::vector<std::vector<double>>): Each edge is [source, target, feature].
 *   Indices are 0‑based; the number of source and target nodes may differ.
 * - N (int): Number of nodes considered (size of node similarity matrices).
 * - M (int): Number of edges considered; used for condensed indexing (1..M).
 * - similarity_score (int): Node similarity metric selector:
 *     0 – Tanimoto, 1 – Cosine, 2 – Jaccard probability,
 *     3 – Hellinger, 4 – Pearson, 5 – Weighted Jaccard.
 *
 * core Class Methods:
 * -------------------
 * - fit_linksim_condense_matrix(): Compute condensed link similarity (size M*(M-1)/2).
 * - fit_linksim_edgelist(): Compute link‑to‑link similarities as an edge list.
 * - get_linksim_condense_matrix(), get_linksim_edgelist(): Retrieve results.
 * - get_source_matrix(), get_target_matrix(): Retrieve node similarity matrices.
 *
 * Implementation Notes:
 * ---------------------
 * - Edges are indexed 1..M after sorting by (source,target); self‑loops are skipped.
 * - Condensed indexing follows the standard upper‑triangle convention with row_id < col_id.
 * - The code guards against malformed inputs and uses exceptions for invalid states
 *   (e.g., negative sizes, mismatched M, invalid similarity index).
 * - Work is parallelized using std::thread where beneficial.
 *
 * Usage:
 * ------
 * Construct core with (edgelist, N, M, similarity_score), then call fit_linksim_condense_matrix()
 * or fit_linksim_edgelist(), and finally query the results via the getters. The module is exposed
 * to Python through pybind11.
 */

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <execution> // For parallel policies (C++17)
#include <cmath>     // For math operations
#include <algorithm> // For std::max
#include<ctime> // time
#include <mutex>
#include <unordered_map>
#include <limits>
#include <stdexcept>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct EdgeInfo {
    double weight;
    int index; // Edge index from 1 to M (number of edges in the subgraph)
};

// edge_struct should be defined as:
struct edge_struct {
    int link1; // Edge index from 1 to M
    int link2; // Edge index from 1 to M
    double similarity;
    edge_struct(int l1, int l2, double sim) : link1(l1), link2(l2), similarity(sim) {}
};

// Note: only source and target nodes from the subgraph analyzed
struct Edge {
    int source; // Source node index from 0 to N-1
    int target; // Target node index from 0 to N-1
    int index;  // Edge index from 1 to M
};

struct NodeNeighbors {
    std::vector<int> neighbors;         // Neighbor nodes in the subgraph
    std::vector<int> edge_indices;      // Indices of edges in the edge list
};

class core {
    private:
    std::vector<double> linksim_condense_matrix;
    std::vector<edge_struct> linksim_edgelist;
    std::vector<std::vector<double> > source_matrix;
    std::vector<std::vector<double> > target_matrix;
    std::vector<std::vector<double> > edgelist;
    int number_of_nodes;
    int number_of_edges;
    int similarity_index;

    public:   
        core(
            const std::vector<std::vector<double> > edgelist,
            const int N,
            const int M,
            const int similarity_score
        );
        ~core(){};

        // 1. Feature matrix extraction
        // Note: Source and target nodes can be different
        std::vector<std::vector<double> > get_out_feature_matrix_from_edgelist();
        std::vector<std::vector<double> > get_in_feature_matrix_from_edgelist();

        // 2. Sorted edgelist computation
        std::vector<std::vector<double>> compute_sorted_edgelist();

        // 3. Node similarity matrix calculation
        std::vector<std::vector<double> > calculate_nodesim_matrix(std::vector<std::vector<double> >& feature_matrix);

        // 4. Link similarity calculation
        std::vector<double> calculate_linksim_condense_matrix(std::vector<std::vector<double>> &sorted_edgelist);
        std::vector<edge_struct> calculate_linksim_edgelist(std::vector<std::vector<double>> &sorted_edgelist);

        // 5. Fit functions
        void fit_linksim_condense_matrix();
        void fit_linksim_edgelist();

        // 6. Getters
        std::vector<double> get_linksim_condense_matrix();
        std::vector<std::vector<double> > get_linksim_edgelist();
        std::vector<std::vector<double> > get_source_matrix();
        std::vector<std::vector<double> > get_target_matrix();

        // 7. Similarity functions
        double similarity_map_function(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
        double tanimoto_coefficient_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
        double cosine_similarity_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
        double jaccard_probability_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
        double hellinger_similarity_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int&jj);
        double pearson_correlation_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
        double weighted_jaccard_graph(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj);
};

core::core(
	const std::vector<std::vector<double>> edgelist,
	const int N,
	const int M,
	const int similarity_score
){
    this->edgelist = edgelist;
	number_of_nodes = N;
	number_of_edges = M;
	similarity_index= similarity_score;
}

std::vector<std::vector<double>> core::get_out_feature_matrix_from_edgelist() {
    // Find max source and target indices
    int max_source = -1, max_target = -1;
    for (const auto& edge : edgelist) {
        if (edge.size() < 2) continue;
        max_source = std::max(max_source, static_cast<int>(edge[0]));
        max_target = std::max(max_target, static_cast<int>(edge[1]));
    }
    int S = max_source + 1;
    int T = max_target + 1;

    // Initialize S x T matrix with zeros
    std::vector<std::vector<double>> out_feature_matrix(S, std::vector<double>(T, 0.0));

    // Parallelize filling the out_feature_matrix
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (edgelist.size() + num_threads - 1) / num_threads;

    auto fill_chunk = [&](int start, int end) {
        for (int idx = start; idx < end; ++idx) {
            const auto& edge = edgelist[idx];
            if (edge.size() < 3) continue;
            int source = static_cast<int>(edge[0]);
            int target = static_cast<int>(edge[1]);
            double feature = edge[2];
            if (source >= 0 && source < S && target >= 0 && target < T) {
                out_feature_matrix[source][target] = feature;
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)edgelist.size());
        threads.emplace_back(fill_chunk, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    return out_feature_matrix;
}

std::vector<std::vector<double>> core::get_in_feature_matrix_from_edgelist() {
    // Find max source and target indices
    int max_source = -1, max_target = -1;
    for (const auto& edge : edgelist) {
        if (edge.size() < 2) continue;
        max_source = std::max(max_source, static_cast<int>(edge[0]));
        max_target = std::max(max_target, static_cast<int>(edge[1]));
    }
    int S = max_source + 1;
    int T = max_target + 1;

    // Initialize T x S matrix with zeros
    std::vector<std::vector<double>> in_feature_matrix(T, std::vector<double>(S, 0.0));

    // Parallelize filling the in_feature_matrix
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (edgelist.size() + num_threads - 1) / num_threads;

    auto fill_chunk = [&](int start, int end) {
        for (int idx = start; idx < end; ++idx) {
            const auto& edge = edgelist[idx];
            if (edge.size() < 3) continue;
            int source = static_cast<int>(edge[0]);
            int target = static_cast<int>(edge[1]);
            double feature = edge[2];
            if (source >= 0 && source < S && target >= 0 && target < T) {
                in_feature_matrix[target][source] = feature;
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)edgelist.size());
        threads.emplace_back(fill_chunk, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    return in_feature_matrix;
}

std::vector<std::vector<double>> core::compute_sorted_edgelist() {
    std::vector<std::vector<double>> sorted_edgelist = edgelist;

    // Remove edges with source or target >= number_of_nodes
    sorted_edgelist.erase(
        std::remove_if(
            sorted_edgelist.begin(),
            sorted_edgelist.end(),
            [this](const std::vector<double>& edge) {
                if (edge.size() < 2) return true;
                int source = static_cast<int>(edge[0]);
                int target = static_cast<int>(edge[1]);
                return (source >= number_of_nodes || target >= number_of_nodes);
            }
        ),
        sorted_edgelist.end()
    );

    std::stable_sort(sorted_edgelist.begin(), sorted_edgelist.end(),
        [](const std::vector<double>& a, const std::vector<double>& b) {
            if (a.size() < 2 || b.size() < 2) return false;
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] < b[1];
        }
    );

    // Add edge index column (from 1 to number_of_edges)
    for (size_t i = 0; i < sorted_edgelist.size(); ++i) {
        if (sorted_edgelist[i].size() < 2) continue; // skip malformed edges
        sorted_edgelist[i].push_back(static_cast<double>(i + 1));
    }
    return sorted_edgelist;
}

std::vector<std::vector<double>> core::calculate_nodesim_matrix(
    std::vector<std::vector<double>>& matrix
) {
    std::vector<std::vector<double>> node_sim_matrix(number_of_nodes, std::vector<double>(number_of_nodes, 0.0));

    auto compute_row = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = i; j < number_of_nodes; ++j) {
                if (i == j) continue;
                node_sim_matrix[i][j] = similarity_map_function(matrix[i], matrix[j], i, j);
                node_sim_matrix[j][i] = node_sim_matrix[i][j];
            }
        }
    };

    // Determine the number of threads to use
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (number_of_nodes + num_threads - 1) / num_threads;

    // Spawn threads
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, number_of_nodes);
        threads.emplace_back(compute_row, start, end);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    return node_sim_matrix;
}

// Calculate link similarity edge list (memory efficient)
std::vector<edge_struct> core::calculate_linksim_edgelist(
    std::vector<std::vector<double>>& sorted_edgelist  // Sorted edge list
) {
    if (number_of_nodes == 0 || number_of_edges <= 0) {
        throw std::invalid_argument("Matrix is empty or number_of_edges is invalid.");
    }

    if (sorted_edgelist.size() != number_of_edges) {
        throw std::invalid_argument("sorted_edgelist size does not match number_of_edges.");
    }

    std::vector<Edge> edge_list;
    std::unordered_map<int, NodeNeighbors> out_neighbors, in_neighbors;
    std::mutex edge_list_mutex, out_neighbors_mutex, in_neighbors_mutex;

    // Step 1: Create edge list and populate neighbor maps from sorted_edgelist
    edge_list.reserve(number_of_edges);  // Reserving memory for edge_list

    auto process_rows_step1 = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const auto& edge = sorted_edgelist[i];
            if (edge.size() < 4) continue;
            int source = static_cast<int>(edge[0]);
            int target = static_cast<int>(edge[1]);
            int edge_index = static_cast<int>(edge[3]);
            if (source == target) continue; // Skip self-loops

            {
                std::lock_guard<std::mutex> lock(edge_list_mutex);
                edge_list.push_back({source, target, edge_index});
            }
            {
                std::lock_guard<std::mutex> lock(out_neighbors_mutex);
                out_neighbors[source].neighbors.push_back(target);
                out_neighbors[source].edge_indices.push_back(edge_index);
            }
            {
                std::lock_guard<std::mutex> lock(in_neighbors_mutex);
                in_neighbors[target].neighbors.push_back(source);
                in_neighbors[target].edge_indices.push_back(edge_index);
            }
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (number_of_edges + num_threads - 1) / num_threads;  // Divide edges into chunks
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)sorted_edgelist.size());
        threads.emplace_back(process_rows_step1, start, end);
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Step 2: Initialize link similarity edge list
	std::vector<edge_struct>  link_similarity_edgelist;
    std::mutex edgelist_mutex;

    // Step 3: Compute link similarities and fill edge list
    auto process_edges_step3 = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const auto& edge = edge_list[i];
            int source = edge.source;
            int target = edge.target;
            int row_id = edge.index;

            // Out-neighbors (same row)
            for (size_t k = 0; k < out_neighbors[source].neighbors.size(); ++k) {
                int neighbor = out_neighbors[source].neighbors[k];
                int col_id = out_neighbors[source].edge_indices[k];
                if (col_id <= row_id) continue;
                double sim = target_matrix[target][neighbor];
                std::lock_guard<std::mutex> lock(edgelist_mutex);
                link_similarity_edgelist.emplace_back(row_id, col_id, sim);
            }
            // In-neighbors (same column)
            for (size_t k = 0; k < in_neighbors[target].neighbors.size(); ++k) {
                int neighbor = in_neighbors[target].neighbors[k];
                int col_id = in_neighbors[target].edge_indices[k];
                if (col_id <= row_id) continue;
				double sim = source_matrix[source][neighbor];
				std::lock_guard<std::mutex> lock(edgelist_mutex);
				link_similarity_edgelist.emplace_back(row_id, col_id, sim);
            }
        }
    };

    threads.clear();
    chunk_size = (edge_list.size() + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)edge_list.size());
        threads.emplace_back(process_edges_step3, start, end);
    }
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }

	// Find the number of distinct elements in the first two columns (link1, link2)
	int max_id = 0;
	for (const auto& edge : link_similarity_edgelist) {
		max_id = std::max(max_id, std::max(edge.link1, edge.link2));
	}

    return link_similarity_edgelist;
}

std::vector<double> core::calculate_linksim_condense_matrix(
    std::vector<std::vector<double>>& sorted_edgelist
) {
    if (number_of_nodes == 0 || number_of_edges <= 0) {
        throw std::invalid_argument("Matrix is empty or number_of_edges is invalid.");
    }

    if (sorted_edgelist.size() != number_of_edges) {
        throw std::invalid_argument("sorted_edgelist size does not match number_of_edges.");
    }

    std::vector<Edge> edge_list;
    std::unordered_map<int, NodeNeighbors> out_neighbors, in_neighbors;
    std::mutex edge_list_mutex, out_neighbors_mutex, in_neighbors_mutex;

    // Step 1: Create edge list and populate neighbor maps from sorted_edgelist
    edge_list.reserve(number_of_edges);  // Reserving memory for edge_list

    auto process_rows_step1 = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const auto& edge = sorted_edgelist[i];
            if (edge.size() < 4) continue;
            int source = static_cast<int>(edge[0]);
            int target = static_cast<int>(edge[1]);
            int edge_index = static_cast<int>(edge[3]);
            if (source == target) continue; // Skip self-loops

            {
                std::lock_guard<std::mutex> lock(edge_list_mutex);
                edge_list.push_back({source, target, edge_index});
            }
            {
                std::lock_guard<std::mutex> lock(out_neighbors_mutex);
                out_neighbors[source].neighbors.push_back(target);
                out_neighbors[source].edge_indices.push_back(edge_index);
            }
            {
                std::lock_guard<std::mutex> lock(in_neighbors_mutex);
                in_neighbors[target].neighbors.push_back(source);
                in_neighbors[target].edge_indices.push_back(edge_index);
            }
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (number_of_edges + num_threads - 1) / num_threads;  // Divide edges into chunks
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)sorted_edgelist.size());
        threads.emplace_back(process_rows_step1, start, end);
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Step 2: Initialize link similarity matrix
    int t = (int)((number_of_edges - 1.0) * number_of_edges / 2.0);
    std::vector<double> link_similarity_condense_matrix(t, 0.0);

    // Step 3: Compute link similarities using the edge list
    auto process_edges_step3 = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const auto& edge = edge_list[i];
            int source = edge.source;
            int target = edge.target;
            int row_id = edge.index;

            // Process out-neighbors (same row)
            for (size_t k = 0; k < out_neighbors[source].neighbors.size(); ++k) {
                int neighbor = out_neighbors[source].neighbors[k];
                int col_id = out_neighbors[source].edge_indices[k];
				if (col_id <= row_id) continue; // Avoid duplicates
                int ri = row_id - 1; // zero-based
                int cj = col_id - 1; // zero-based
                int idx = number_of_edges * ri - (ri * (ri + 1)) / 2 + (cj - ri - 1);
                link_similarity_condense_matrix[idx] = target_matrix[target][neighbor];
            }

            // Process in-neighbors (same column)
            for (size_t k = 0; k < in_neighbors[target].neighbors.size(); ++k) {
                int neighbor = in_neighbors[target].neighbors[k];
                int col_id = in_neighbors[target].edge_indices[k];
				if (col_id <= row_id) continue; // Avoid duplicates
                int ri = row_id - 1; // zero-based
                int cj = col_id - 1; // zero-based
                int idx = number_of_edges * ri - (ri * (ri + 1)) / 2 + (cj - ri - 1);
                link_similarity_condense_matrix[idx] = source_matrix[source][neighbor];
            }
        }
    };

    // Launch threads for Step 3
    threads.clear();
    chunk_size = (edge_list.size() + num_threads - 1) / num_threads;  // Divide edges into chunks
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)edge_list.size());
        threads.emplace_back(process_edges_step3, start, end);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return link_similarity_condense_matrix;
}

void core::fit_linksim_condense_matrix() {
    // Step 1: Compute feature matrices
    // Out-feature matrix (size S x T)
    std::vector<std::vector<double>> out_feature_matrix = get_out_feature_matrix_from_edgelist();
    // In-feature matrix (size T x S)
    std::vector<std::vector<double>> in_feature_matrix = get_in_feature_matrix_from_edgelist();

    // Step 2: Compute node similarity matrices
    // Source node similarity matrix (size N x N)
	source_matrix = calculate_nodesim_matrix(out_feature_matrix);
    // Target node similarity matrix (size N x N)
	target_matrix = calculate_nodesim_matrix(in_feature_matrix);

    // Step 3: Compute link similarity condensed matrix
    // sorted edgelist (size M x 4)
    std::vector<std::vector<double>> sorted_edgelist = compute_sorted_edgelist();
	linksim_condense_matrix = calculate_linksim_condense_matrix(sorted_edgelist);
}

void core::fit_linksim_edgelist() {
    // Step 1: Compute feature matrices
    // Out-feature matrix (size S x T)
    std::vector<std::vector<double>> out_feature_matrix = get_out_feature_matrix_from_edgelist();
    // In-feature matrix (size T x S)
    std::vector<std::vector<double>> in_feature_matrix = get_in_feature_matrix_from_edgelist();

    // Step 2: Compute node similarity matrices
    // Source node similarity matrix (size N x N)
	source_matrix = calculate_nodesim_matrix(out_feature_matrix);
    // Target node similarity matrix (size N x N)
	target_matrix = calculate_nodesim_matrix(in_feature_matrix);

    // Step 3: Compute link similarity edge list
    // sorted edgelist (size M x 4)
    std::vector<std::vector<double>> sorted_edgelist = compute_sorted_edgelist();
	linksim_edgelist = calculate_linksim_edgelist(sorted_edgelist);
}

std::vector<double> core::get_linksim_condense_matrix() {
	return linksim_condense_matrix;
}

std::vector<std::vector<double>> core::get_linksim_edgelist() {
    // Each edge_struct has id1, id2, sim12
    // Return as a vector of vectors: {{id1, id2, sim12}, ...}
    std::vector<std::vector<double>> result;
    result.reserve(linksim_edgelist.size());
    for (const auto& edge : linksim_edgelist) {
        result.push_back({
            static_cast<double>(edge.link1),
            static_cast<double>(edge.link2),
            edge.similarity
        });
    }
    return result;
}

std::vector<std::vector<double> > core::get_source_matrix() {
	return source_matrix;
}

std::vector<std::vector<double> > core::get_target_matrix() {
	return target_matrix;
}

double core::similarity_map_function(std::vector<double> &u, std::vector<double> &v, int &ii, int &jj) {
	if (similarity_index == 0) {
		return tanimoto_coefficient_graph(u, v, ii, jj);
	}
	else if (similarity_index == 1) {
		return cosine_similarity_graph(u, v, ii, jj);
	}
	else if (similarity_index == 2) {
		return jaccard_probability_graph(u, v, ii, jj);
	}
	else if (similarity_index == 3) {
		return hellinger_similarity_graph(u, v, ii, jj);
	}
	else if (similarity_index == 4) {
		return pearson_correlation_graph(u, v, ii, jj);
	}
	else if (similarity_index == 5) {
		return weighted_jaccard_graph(u, v, ii, jj);
	}
    else {
        throw std::range_error("Similarity index must be an integer from 0 to 5");
    }
}

double core::tanimoto_coefficient_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
    for (int i=0; i < N; i++) {
        uu += u[i] * u[i];
        vv += v[i] * v[i];
        if (i == ii || i == jj) continue;
        uv += u[i] * v[i];
    }
	if (ii < N && jj < N) {
		uv += u[jj] * v[ii];
		uv += u[ii] * v[jj]; 
	}

    if (uu <= 0 && vv <= 0 && uv <= 0) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");

	return uv / (uu + vv - uv);
}

double core::cosine_similarity_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;

    for (int i=0; i < N; i++) {
        uu += u[i] * u[i];
        vv += v[i] * v[i];
        if (i == ii || i == jj) continue;
        uv += u[i] * v[i];
    }

	if (ii < N && jj < N) {
		uv += u[jj] * v[ii];
		uv += u[ii] * v[jj]; 
	}

    if (uu <= 0 || vv <= 0) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");

	return uv / (sqrt(uu * vv));
}

double core::pearson_correlation_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0., mu=0., mv=0.;

    for (int i=0; i < N; i++) {
        mu += u[i];
        mv += v[i];
        uu += u[i] * u[i];
        vv += v[i] * v[i];
        if (i == ii || i == jj) continue;
        uv += u[i] * v[i];
    }

	mu /= N;
	mv /= N;

	if (ii < N && jj < N) {
		uv += u[jj] * v[ii];
		uv += u[ii] * v[jj]; 
	}

	uv /= N;
	uu /= N;
	vv /= N;
	uu -= pow(mu, 2);
	vv -= pow(mv, 2);
	if (uu <= 0 || vv <= 0) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");

	return (uv - mu * mv) / (sqrt( uu * vv));
}

double core::weighted_jaccard_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double maximus=0., minimum=0.;
	for (int i=0; i < N; i++) {
		if ( i == ii || i == jj) continue;
		minimum += std::min(u[i], v[i]);
		maximus += std::max(u[i], v[i]);
	}

	minimum += std::min(u[jj], v[ii]);
	maximus += std::max(u[jj], v[ii]);

	minimum += std::min(u[ii], v[jj]);
	maximus += std::max(u[ii], v[jj]);

	if (minimum == 0 && maximus == 0) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");


	return minimum / maximus;
}

double core::hellinger_similarity_graph(
    std::vector<double> &fu, std::vector<double> &fv, int &iu, int &iv
) {
    int N = fu.size();
    double pu = 0.0, pv = 0.0;
    double s = 0.0, maxp = -std::numeric_limits<double>::infinity();

    std::vector<double> ppu(N, 0.0), ppv(N, 0.0), peff(N, 0.0);
    std::vector<bool> possible(N, false);

    // Compute normalizers and rearranged vectors
    for (int j = 0; j < N; ++j) {
        pu += fu[j];
        pv += fv[j];
    }

    if (pu <= 0 || pv <= 0) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");

    // Rearranged to handle iu and iv explicitly
    int k = 0;
    for (int j = 0; j < N; ++j) {
        if (j == iu || j == iv) continue;
        ppu[k] = fu[j];
        ppv[k] = fv[j];
        k++;
    }
    // Assign the special indices
    ppu[N-2] = fu[iu];
    ppu[N-1] = fu[iv];
    ppv[N-2] = fv[iv];
    ppv[N-1] = fv[iu];

    for (int j = 0; j < N; ++j) {
        if (ppu[j] > 0 && ppv[j] > 0) {
            peff[j] = 0.5 * (log(ppu[j]) + log(ppv[j]) - log(pu) - log(pv));
            possible[j] = true;
            if (peff[j] > maxp) maxp = peff[j];
        }
    }

    if (maxp == -std::numeric_limits<double>::infinity()) throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");

    for (int j = 0; j < N; ++j) {
        if (possible[j]) {
            s += exp(peff[j] - maxp);
        }
    }

    return s * exp(maxp);  // Approximation of the Bhattacharyya coefficient
}

double core::jaccard_probability_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj)
{
	int N = u.size();
	double jacp = 0;
	double p;
    for (int i=0; i < N; i++){
        if ((u[i] > 0 && v[i] > 0) && (i != ii && i != jj)){
            p = 0;
            for (int j=0; j < N; j++) {
                if (j == ii || j == jj) continue;
                p += std::max(u[j]/u[i], v[j]/v[i]);
            }
            p += std::max(u[jj]/u[i], v[ii]/v[i]);
            if (p != 0)
                jacp += 1 / p;
            else
                std::cout << "Vectors with indices " << ii << " and " << jj << " are both zero\n";
        }
    }
	if (u[jj] > 0 && v[ii] > 0) {
		p = 0;
		for (int j=0; j < N; j++) {
			if (j == ii || j == jj) continue;
			p += std::max(u[j]/u[jj], v[j]/v[ii]);
		}
		p += 1;
		if (p != 0)
			jacp += 1 / p;
		else
			std::cout << "Vectors with indices " << ii << " and " << jj << " are both zero\n";
	}
	return jacp;
}

PYBIND11_MODULE(linksim_cpp, m) {
    py::class_<core>(m, "core")
        .def(
            py::init<
            const std::vector<std::vector<double>>,
			const int,
			const int,
			const int
          >()
        )
		.def("fit_linksim_condense_matrix", &core::fit_linksim_condense_matrix)
        .def("get_linksim_condense_matrix", &core::get_linksim_condense_matrix)
		.def("fit_linksim_edgelist", &core::fit_linksim_edgelist)
		.def("get_linksim_edgelist", &core::get_linksim_edgelist)	
        .def("get_source_matrix", &core::get_source_matrix)
		.def("get_target_matrix", &core::get_target_matrix);
}
