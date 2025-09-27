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
 * Multithreading can be toggled on when desirable to accelerate the computation.
 *
 * Main Components:
 * ----------------
 * - core class: Encapsulates the computation pipeline. It extracts sparse feature maps from an
 *   edge list, computes node–node similarities, sorts/indexes the edges, and produces either
 *   a condensed link‑similarity matrix or a link‑to‑link edgelist.
 * - edge_struct: Lightweight struct that stores a pair of link indices and their similarity.
 * - Internal helpers (Edge, NodeNeighbors): Support efficient neighborhood traversal.
 *
 * core Class Parameters:
 * ----------------------
 * - N (int): Number of nodes considered (size of node similarity matrices).
 * - M (int): Number of edges considered; used for condensed indexing (1..M).
 * - edgelist (std::vector<std::vector<double>>): Each edge is [source, target, feature].
 *   Indices are 0‑based; the number of source and target nodes may differ.
 *   During preprocessing the library builds sparse source/target feature maps and tracks the
 *   maximum node index to support similarity metrics that need the full dimensionality.
 * - similarity_score (int): Node similarity metric selector:
 *     0 – Tanimoto, 1 – Cosine, 2 – Jaccard probability,
 *     3 – Hellinger, 4 – Pearson, 5 – Weighted Jaccard.
 * - undirected (bool, default false): Treat the graph as undirected (symmetric node similarities).
 * - use_parallel (bool, default true): Enable multithreaded edge processing.
 * - flat (bool, default false): Allow zero feature vectors; similarities return 0 instead of
 *   throwing when at least one vector is all zeros.
 * - verbose (int, optional): Verbosity level (0 silent, 1 workflow milestones, 2 step announcements).
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
 * Construct core with (edgelist, N, M, similarity_score, use_parallel, flat), then call
 * fit_linksim_condense_matrix() or fit_linksim_edgelist(), and finally query the results via the
 * getters. Pass flat=true to map zero feature vectors to similarity 0. The module is exposed to
 * Python through pybind11.
*/

#include <iostream>
#include <vector>
#include <string>
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
#include <exception>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using NodeFeatureMap = std::unordered_map<int, float>;
using FeatureMap = std::unordered_map<int, NodeFeatureMap>;

static inline double weight_or_zero(const NodeFeatureMap &features, int key) {
    auto it = features.find(key);
    return (it != features.end()) ? static_cast<double>(it->second) : 0.0;
}

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
    bool undirected;
    bool use_parallel;
    bool flat_mode;
    int verbose;
    int max_dim;

    void workflow(int level, const std::string &message);

    public:   
        core(
            const int N,
            const int M,
            const std::vector<std::vector<double> > edgelist,
            const int similarity_score,
            const bool undirected = false,
            const bool enable_parallel = true,
            const bool flat = false,
            const int verbose_level = 0
        );
        ~core(){};

        // 1. Feature maps extraction
        // Note: Source and target nodes can be different
        FeatureMap get_out_feature_map_from_edgelist();
        FeatureMap get_in_feature_map_from_edgelist();

        // 2. Sorted edgelist computation
        std::vector<std::vector<double>> compute_sorted_edgelist();

        // 3. Node similarity matrix calculation
        std::vector<std::vector<double> > calculate_nodesim_matrix(const FeatureMap& feature_map);

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
        double similarity_map_function(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double tanimoto_coefficient_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double cosine_similarity_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double jaccard_probability_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double bhattacharyya_coefficient_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double pearson_correlation_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
        double weighted_jaccard_graph(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj);
};

void core::workflow(int level, const std::string &message) {
    if (verbose >= level) {
        std::cout << "[linksim] " << message << std::endl;
    }
}

core::core(
	const int N,
	const int M,
	const std::vector<std::vector<double>> edgelist,
	const int similarity_score,
    const bool undirected,
	const bool enable_parallel,
	const bool flat,
    const int verbose_level
){
	number_of_nodes = N;
	number_of_edges = M;
    this->edgelist = edgelist;
	similarity_index = similarity_score;
    this->undirected = undirected;
    use_parallel = enable_parallel;
    flat_mode = flat;
    max_dim = 0;
    if (verbose_level < 0 || verbose_level > 2) {
        std::cerr << "Warning: Verbose level " << verbose_level << " out of range [0,2]. Clamping to nearest bound.\n";
        verbose = (verbose_level < 0) ? 0 : 2;
    } else {
        verbose = verbose_level;
    }

    if (verbose >= 1) {
        std::cout << "[linksim] Initialized with N=" << number_of_nodes
                  << ", M=" << number_of_edges
                  << ", verbose=" << verbose << std::endl;
    }
}

void core::fit_linksim_condense_matrix() {
    workflow(1, "Starting fit_linksim_condense_matrix()");
    workflow(2, "Step 1: computing feature maps");
    // Step 1: Compute sparse feature maps
    FeatureMap out_feature_map = get_out_feature_map_from_edgelist();
    FeatureMap in_feature_map = get_in_feature_map_from_edgelist();

    workflow(2, "Step 2: computing node similarity matrices");
    // Step 2: Compute node similarity matrices
    source_matrix = calculate_nodesim_matrix(out_feature_map);
    target_matrix = calculate_nodesim_matrix(in_feature_map);

    workflow(2, "Step 3: computing condensed link similarity");
    // Step 3: Compute link similarity condensed matrix
    std::vector<std::vector<double>> sorted_edgelist = compute_sorted_edgelist();
    linksim_condense_matrix = calculate_linksim_condense_matrix(sorted_edgelist);
    workflow(1, "Completed fit_linksim_condense_matrix()");
}

void core::fit_linksim_edgelist() {
    workflow(1, "Starting fit_linksim_edgelist()");
    workflow(2, "Step 1: computing feature maps");
    // Step 1: Compute sparse feature maps
    // Out-feature map: source -> {target: weight}
    FeatureMap out_feature_map = get_out_feature_map_from_edgelist();
    // In-feature map: target -> {source: weight}
    FeatureMap in_feature_map = get_in_feature_map_from_edgelist();

    workflow(2, "Step 2: computing node similarity matrices");
    // Step 2: Compute node similarity matrices
    // Source node similarity matrix (size N x N)
	source_matrix = calculate_nodesim_matrix(out_feature_map);
    // Target node similarity matrix (size N x N)
	target_matrix = calculate_nodesim_matrix(in_feature_map);

    workflow(2, "Step 3: computing link similarity edgelist");
    // Step 3: Compute link similarity edge list
    // sorted edgelist (size M x 4)
    std::vector<std::vector<double>> sorted_edgelist = compute_sorted_edgelist();
	linksim_edgelist = calculate_linksim_edgelist(sorted_edgelist);
    workflow(1, "Completed fit_linksim_edgelist()");
}

FeatureMap core::get_out_feature_map_from_edgelist() {
    workflow(2, "Preparing out-feature map from edgelist");
    int max_index = -1;
    for (const auto& edge : edgelist) {
        if (edge.size() < 2) continue;
        max_index = std::max(max_index, static_cast<int>(edge[0]));
        max_index = std::max(max_index, static_cast<int>(edge[1]));
    }
    int candidate_dimension = (max_index >= 0)
        ? std::max(max_index + 1, number_of_nodes)
        : number_of_nodes;
    max_dim = std::max(max_dim, candidate_dimension);

    FeatureMap out_feature_map;
    for (const auto& edge : edgelist) {
        if (edge.size() < 3) continue;
        int source = static_cast<int>(edge[0]);
        int target = static_cast<int>(edge[1]);
        float feature = static_cast<float>(edge[2]);
        if (source < 0 || target < 0) continue;

        out_feature_map[source][target] = feature;
        if (undirected) {
            out_feature_map[target][source] = feature;
        }
    }

    return out_feature_map;
}

FeatureMap core::get_in_feature_map_from_edgelist() {
    workflow(2, "Preparing in-feature map from edgelist");
    int max_index = -1;
    for (const auto& edge : edgelist) {
        if (edge.size() < 2) continue;
        max_index = std::max(max_index, static_cast<int>(edge[0]));
        max_index = std::max(max_index, static_cast<int>(edge[1]));
    }
    int candidate_dimension = (max_index >= 0)
        ? std::max(max_index + 1, number_of_nodes)
        : number_of_nodes;
    max_dim = std::max(max_dim, candidate_dimension);

    FeatureMap in_feature_map;
    for (const auto& edge : edgelist) {
        if (edge.size() < 3) continue;
        int source = static_cast<int>(edge[0]);
        int target = static_cast<int>(edge[1]);
        float feature = static_cast<float>(edge[2]);
        if (source < 0 || target < 0) continue;

        in_feature_map[target][source] = feature;
        if (undirected) {
            in_feature_map[source][target] = feature;
        }
    }

    return in_feature_map;
}

std::vector<std::vector<double>> core::compute_sorted_edgelist() {
    workflow(2, "Computing sorted edgelist");
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
    const FeatureMap& feature_map
) {
    workflow(2, "Calculating node similarity matrix");
    std::vector<std::vector<double>> node_sim_matrix(number_of_nodes, std::vector<double>(number_of_nodes, 0.0));

    const NodeFeatureMap empty_map;
    std::exception_ptr thread_exception;
    std::mutex exception_mutex;

    auto compute_row = [&](int start, int end) {
        try {
            for (int i = start; i < end; ++i) {
                const NodeFeatureMap* u_ptr = &empty_map;
                auto u_it = feature_map.find(i);
                if (u_it != feature_map.end()) {
                    u_ptr = &(u_it->second);
                }

                for (int j = i + 1; j < number_of_nodes; ++j) {
                    const NodeFeatureMap* v_ptr = &empty_map;
                    auto v_it = feature_map.find(j);
                    if (v_it != feature_map.end()) {
                        v_ptr = &(v_it->second);
                    }

                    double similarity = similarity_map_function(*u_ptr, *v_ptr, i, j);
                    node_sim_matrix[i][j] = similarity;
                    node_sim_matrix[j][i] = similarity;
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!thread_exception) thread_exception = std::current_exception();
        }
    };

    int num_threads = use_parallel ? std::max(1u, std::thread::hardware_concurrency()) : 1;
    int chunk_size = (number_of_nodes + num_threads - 1) / num_threads;

    if (use_parallel && num_threads > 1 && chunk_size > 0) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, number_of_nodes);
            if (start >= end) continue;
            threads.emplace_back(compute_row, start, end);
        }
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    } else {
        compute_row(0, number_of_nodes);
    }

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    return node_sim_matrix;
}

// Calculate link similarity edge list (memory efficient)
std::vector<edge_struct> core::calculate_linksim_edgelist(
    std::vector<std::vector<double>>& sorted_edgelist  // Sorted edge list
) {
    workflow(2, "Calculating link similarity edge list");
    if (number_of_nodes == 0 || number_of_edges <= 0) {
        throw std::invalid_argument("Matrix is empty or number_of_edges is invalid.");
    }

    if (sorted_edgelist.size() != number_of_edges) {
        throw std::invalid_argument("sorted_edgelist size does not match number_of_edges.");
    }

    std::vector<Edge> edge_list;
    std::unordered_map<int, NodeNeighbors> out_neighbors, in_neighbors;
    std::mutex edge_list_mutex, out_neighbors_mutex, in_neighbors_mutex;

    workflow(2, "Link similarity edge list: building neighbor structures");
    // Step 1: Create edge list and populate neighbor maps from sorted_edgelist
    edge_list.reserve(number_of_edges);  // Reserving memory for edge_list

    std::exception_ptr stage1_exception;
    std::mutex stage1_exception_mutex;

    auto process_rows_step1 = [&](int start, int end) {
        try {
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

                    if (undirected) {
                        out_neighbors[target].neighbors.push_back(source);
                        out_neighbors[target].edge_indices.push_back(edge_index);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(in_neighbors_mutex);
                    in_neighbors[target].neighbors.push_back(source);
                    in_neighbors[target].edge_indices.push_back(edge_index);

                    if (undirected) {
                        in_neighbors[source].neighbors.push_back(target);
                        in_neighbors[source].edge_indices.push_back(edge_index);
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(stage1_exception_mutex);
            if (!stage1_exception) stage1_exception = std::current_exception();
        }
    };

    int num_threads = use_parallel ? std::max(1u, std::thread::hardware_concurrency()) : 1;
    std::vector<std::thread> threads;
    int chunk_size = (number_of_edges + num_threads - 1) / num_threads;  // Divide edges into chunks

    if (use_parallel && num_threads > 1 && chunk_size > 0) {
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, static_cast<int>(sorted_edgelist.size()));
            if (start >= end) continue;
            threads.emplace_back(process_rows_step1, start, end);
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } else {
        process_rows_step1(0, static_cast<int>(sorted_edgelist.size()));
    }

    if (stage1_exception) {
        std::rethrow_exception(stage1_exception);
    }

    // Step 2: Initialize link similarity edge list
	std::vector<edge_struct>  link_similarity_edgelist;
    std::mutex edgelist_mutex;

    workflow(2, "Link similarity edge list: computing pairwise similarities");
    // Step 3: Compute link similarities and fill edge list
    std::exception_ptr stage3_exception;
    std::mutex stage3_exception_mutex;

    auto process_edges_step3 = [&](int start, int end) {
        try {
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
        } catch (...) {
            std::lock_guard<std::mutex> lock(stage3_exception_mutex);
            if (!stage3_exception) stage3_exception = std::current_exception();
        }
    };

    threads.clear();
    chunk_size = (edge_list.size() + num_threads - 1) / num_threads;
    if (use_parallel && num_threads > 1 && chunk_size > 0) {
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, static_cast<int>(edge_list.size()));
            if (start >= end) continue;
            threads.emplace_back(process_edges_step3, start, end);
        }
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    } else {
        process_edges_step3(0, static_cast<int>(edge_list.size()));
    }

    if (stage3_exception) {
        std::rethrow_exception(stage3_exception);
    }

	// Find the number of distinct elements in the first two columns (link1, link2)
	int max_id = 0;
	for (const auto& edge : link_similarity_edgelist) {
		max_id = std::max(max_id, std::max(edge.link1, edge.link2));
	}

    workflow(2, "Link similarity edge list computation finished");

    return link_similarity_edgelist;
}

std::vector<double> core::calculate_linksim_condense_matrix(
    std::vector<std::vector<double>>& sorted_edgelist
) {
    workflow(2, "Calculating condensed link similarity matrix");
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

    std::exception_ptr stage1_exception;
    std::mutex stage1_exception_mutex;

    auto process_rows_step1 = [&](int start, int end) {
        try {
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

                    if (undirected) {
                        out_neighbors[target].neighbors.push_back(source);
                        out_neighbors[target].edge_indices.push_back(edge_index);
                    }
                }
                {
                     std::lock_guard<std::mutex> lock(in_neighbors_mutex);
                    in_neighbors[target].neighbors.push_back(source);
                    in_neighbors[target].edge_indices.push_back(edge_index);

                    if (undirected) {
                        in_neighbors[source].neighbors.push_back(target);
                        in_neighbors[source].edge_indices.push_back(edge_index);
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(stage1_exception_mutex);
            if (!stage1_exception) stage1_exception = std::current_exception();
        }
    };

    int num_threads = use_parallel ? std::max(1u, std::thread::hardware_concurrency()) : 1;
    std::vector<std::thread> threads;
    int chunk_size = (number_of_edges + num_threads - 1) / num_threads;  // Divide edges into chunks

    if (use_parallel && num_threads > 1 && chunk_size > 0) {
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, static_cast<int>(sorted_edgelist.size()));
            if (start >= end) continue;
            threads.emplace_back(process_rows_step1, start, end);
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } else {
        process_rows_step1(0, static_cast<int>(sorted_edgelist.size()));
    }

    if (stage1_exception) {
        std::rethrow_exception(stage1_exception);
    }

    workflow(2, "Condensed similarity: initializing output storage");
    // Step 2: Initialize link similarity matrix
    int t = (int)((number_of_edges - 1.0) * number_of_edges / 2.0);
    std::vector<double> link_similarity_condense_matrix(t, 0.0);

    workflow(2, "Condensed similarity: computing pairwise similarities");
    // Step 3: Compute link similarities using the edge list
    std::exception_ptr stage3_exception;
    std::mutex stage3_exception_mutex;

    auto process_edges_step3 = [&](int start, int end) {
        try {
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
        } catch (...) {
            std::lock_guard<std::mutex> lock(stage3_exception_mutex);
            if (!stage3_exception) stage3_exception = std::current_exception();
        }
    };

    threads.clear();
    chunk_size = (edge_list.size() + num_threads - 1) / num_threads;  // Divide edges into chunks
    if (use_parallel && num_threads > 1 && chunk_size > 0) {
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, static_cast<int>(edge_list.size()));
            if (start >= end) continue;
            threads.emplace_back(process_edges_step3, start, end);
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } else {
        process_edges_step3(0, static_cast<int>(edge_list.size()));
    }

    if (stage3_exception) {
        std::rethrow_exception(stage3_exception);
    }

    workflow(2, "Condensed similarity computation finished");

    return link_similarity_condense_matrix;
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

double core::similarity_map_function(const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj) {
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
		return bhattacharyya_coefficient_graph(u, v, ii, jj);
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
	const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj
) {
	double uv = 0.0;
	double uu = 0.0;
	double vv = 0.0;

    for (const auto& entry : u) {
        double value = static_cast<double>(entry.second);
        uu += value * value;
    }
    for (const auto& entry : v) {
        double value = static_cast<double>(entry.second);
        vv += value * value;
    }

    const NodeFeatureMap* smaller = &u;
    const NodeFeatureMap* larger = &v;
    if (v.size() < u.size()) {
        smaller = &v;
        larger = &u;
    }

    for (const auto& entry : *smaller) {
        int neighbor = entry.first;
        if (neighbor == ii || neighbor == jj) continue;
        auto it = larger->find(neighbor);
        if (it != larger->end()) {
            uv += static_cast<double>(entry.second) * static_cast<double>(it->second);
        }
    }

    uv += weight_or_zero(u, jj) * weight_or_zero(v, ii); // mutual interaction
    uv += weight_or_zero(u, ii) * weight_or_zero(v, jj); // self interaction

    if (uu <= 0 && vv <= 0 && uv <= 0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");
    }

    double result = uv / (uu + vv - uv);
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}

double core::cosine_similarity_graph(
	const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj
) {
	double uv = 0.0;
	double uu = 0.0;
	double vv = 0.0;

    for (const auto& entry : u) {
        double value = static_cast<double>(entry.second);
        uu += value * value;
    }
    for (const auto& entry : v) {
        double value = static_cast<double>(entry.second);
        vv += value * value;
    }

    const NodeFeatureMap* smaller = &u;
    const NodeFeatureMap* larger = &v;
    if (v.size() < u.size()) {
        smaller = &v;
        larger = &u;
    }

    for (const auto& entry : *smaller) {
        int neighbor = entry.first;
        if (neighbor == ii || neighbor == jj) continue;
        auto it = larger->find(neighbor);
        if (it != larger->end()) {
            uv += static_cast<double>(entry.second) * static_cast<double>(it->second);
        }
    }

    uv += weight_or_zero(u, jj) * weight_or_zero(v, ii); // mutual interaction
    uv += weight_or_zero(u, ii) * weight_or_zero(v, jj); // self interaction

    if (uu <= 0 || vv <= 0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");
    }

    double result = uv / (sqrt(uu * vv));
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}

double core::pearson_correlation_graph(
	const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj
) {
    int dimension = std::max(max_dim, number_of_nodes);
    if (dimension <= 0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("Feature space dimension must be positive for Pearson correlation.");
    }

	double uv = 0.0;
	double uu = 0.0;
	double vv = 0.0;
	double mu = 0.0;
	double mv = 0.0;

    for (const auto& entry : u) {
        double value = static_cast<double>(entry.second);
        mu += value;
        uu += value * value;
    }
    for (const auto& entry : v) {
        double value = static_cast<double>(entry.second);
        mv += value;
        vv += value * value;
    }

    const NodeFeatureMap* smaller = &u;
    const NodeFeatureMap* larger = &v;
    if (v.size() < u.size()) {
        smaller = &v;
        larger = &u;
    }

    for (const auto& entry : *smaller) {
        int neighbor = entry.first;
        if (neighbor == ii || neighbor == jj) continue;
        auto it = larger->find(neighbor);
        if (it != larger->end()) {
            uv += static_cast<double>(entry.second) * static_cast<double>(it->second);
        }
    }

    uv += weight_or_zero(u, jj) * weight_or_zero(v, ii); // mutual interaction
    uv += weight_or_zero(u, ii) * weight_or_zero(v, jj); // self interaction

    double dimension_d = static_cast<double>(dimension);
    double mu_avg = mu / dimension_d;
    double mv_avg = mv / dimension_d;

    uv /= dimension_d;
    uu /= dimension_d;
    vv /= dimension_d;
    uu -= mu_avg * mu_avg;
    vv -= mv_avg * mv_avg;

    if (uu <= 0 || vv <= 0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");
    }

    double result = (uv - mu_avg * mv_avg) / (sqrt(uu * vv));
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}

double core::weighted_jaccard_graph(
	const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj
) {
	double maximus = 0.0;
	double minimum = 0.0;

    auto accumulate_key = [&](int key) {
        if (key == ii || key == jj) return;
        double u_val = weight_or_zero(u, key);
        double v_val = weight_or_zero(v, key);
        minimum += std::min(u_val, v_val);
        maximus += std::max(u_val, v_val);
    };

    for (const auto& entry : u) {
        accumulate_key(entry.first);
    }
    for (const auto& entry : v) {
        if (u.find(entry.first) == u.end()) {
            accumulate_key(entry.first);
        }
    }

    minimum += std::min(weight_or_zero(u, jj), weight_or_zero(v, ii)); // mutual interaction
    maximus += std::max(weight_or_zero(u, jj), weight_or_zero(v, ii));

    minimum += std::min(weight_or_zero(u, ii), weight_or_zero(v, jj)); // self interaction
    maximus += std::max(weight_or_zero(u, ii), weight_or_zero(v, jj));

    if (minimum == 0.0 && maximus == 0.0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");
    }

    double result = minimum / maximus;
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}

double core::bhattacharyya_coefficient_graph(
    const NodeFeatureMap &fu, const NodeFeatureMap &fv, int iu, int iv
) {
    double Zu = 0.0, Zv = 0.0, BC = 0.0; // Bhattacharyya coefficient

    for (const auto& entry : fu) {
        Zu += static_cast<double>(entry.second);
    }
    for (const auto& entry : fv) {
        Zv += static_cast<double>(entry.second);
    }

    if (Zu <= 0.0 || Zv <= 0.0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("One or both feature vectors have all zeros, which is not allowed.");
    }

    const NodeFeatureMap* smaller = &fu;
    const NodeFeatureMap* larger = &fv;
    if (fv.size() < fu.size()) {
        smaller = &fv;
        larger = &fu;
    }

    for (const auto& entry : *smaller) {
        int neighbor = entry.first;
        if (neighbor == iu || neighbor == iv) continue;
        auto it = larger->find(neighbor);
        if (it != larger->end()) {
            double value = sqrt(static_cast<double>(entry.second) * static_cast<double>(it->second) / (Zu * Zv));
            BC += value;
        }
    }

    double mutual = weight_or_zero(fu, iv) * weight_or_zero(fv, iu);
    if (mutual > 0.0) {
        BC += sqrt(mutual / (Zu * Zv));
    }
    double self_interaction = weight_or_zero(fu, iu) * weight_or_zero(fv, iv);
    if (self_interaction > 0.0) {
        BC += sqrt(self_interaction / (Zu * Zv));
    }

    if (BC < 0.0) BC = 0.0;
    if (BC > 1.0) BC = 1.0;
    return BC;
}

double core::jaccard_probability_graph(
	const NodeFeatureMap &u, const NodeFeatureMap &v, int ii, int jj)
{
    int dimension = std::max(max_dim, number_of_nodes);
    if (dimension <= 0) {
        if (flat_mode) return 0.0;
        throw std::invalid_argument("Feature space dimension must be positive for Jaccard probability.");
    }

    std::vector<double> u_vec(dimension, 0.0);
    std::vector<double> v_vec(dimension, 0.0);

    for (const auto& entry : u) {
        if (entry.first >= 0 && entry.first < dimension) {
            u_vec[entry.first] = static_cast<double>(entry.second);
        }
    }
    for (const auto& entry : v) {
        if (entry.first >= 0 && entry.first < dimension) {
            v_vec[entry.first] = static_cast<double>(entry.second);
        }
    }

    auto value_at = [&](const std::vector<double>& vec, int idx) -> double {
        if (idx < 0 || idx >= dimension) return 0.0;
        return vec[idx];
    };

	double jacp = 0.0;
	double p;
    for (int i = 0; i < dimension; i++){
        if ((u_vec[i] > 0 && v_vec[i] > 0) && (i != ii && i != jj)){
            p = 0.0;
            for (int j = 0; j < dimension; j++) {
                if (j == ii || j == jj) continue;
                double denom_u = u_vec[i];
                double denom_v = v_vec[i];
                double term_u = denom_u > 0 ? u_vec[j] / denom_u : 0.0;
                double term_v = denom_v > 0 ? v_vec[j] / denom_v : 0.0;
                p += std::max(term_u, term_v);
            }
            double denom_u = u_vec[i];
            double denom_v = v_vec[i];
            double term_mutual_u = denom_u > 0 ? value_at(u_vec, jj) / denom_u : 0.0;
            double term_mutual_v = denom_v > 0 ? value_at(v_vec, ii) / denom_v : 0.0;
            p += std::max(term_mutual_u, term_mutual_v);
            if (p != 0.0)
                jacp += 1.0 / p;
            else
                std::cout << "Vectors with indices " << ii << " and " << jj << " are both zero\n";
        }
    }
	if (value_at(u_vec, jj) > 0 && value_at(v_vec, ii) > 0) {
		p = 0.0;
		for (int j = 0; j < dimension; j++) {
			if (j == ii || j == jj) continue;
			double denom_u = value_at(u_vec, jj);
			double denom_v = value_at(v_vec, ii);
			double term_u = denom_u > 0 ? u_vec[j] / denom_u : 0.0;
			double term_v = denom_v > 0 ? v_vec[j] / denom_v : 0.0;
			p += std::max(term_u, term_v);
		}
		p += 1.0;
		if (p != 0.0)
			jacp += 1.0 / p;
		else
			std::cout << "Vectors with indices " << ii << " and " << jj << " are both zero\n";
	}
	return jacp;
}

PYBIND11_MODULE(linksim_cpp, m) {
    py::class_<core>(m, "core", py::module_local())
        .def(
            py::init<
                const int,
                const int,
                const std::vector<std::vector<double>>,
                const int,
                const bool,
                const bool,
                const bool,
                const int
            >(),
            py::arg("N"),
            py::arg("M"),
            py::arg("edgelist"),
            py::arg("similarity_score"),
            py::arg("undirected") = false,
            py::arg("use_parallel") = true,
            py::arg("flat") = false,
            py::arg("verbose") = 0
        )
		.def("fit_linksim_condense_matrix", &core::fit_linksim_condense_matrix)
        .def("get_linksim_condense_matrix", &core::get_linksim_condense_matrix)
		.def("fit_linksim_edgelist", &core::fit_linksim_edgelist)
		.def("get_linksim_edgelist", &core::get_linksim_edgelist)	
        .def("get_source_matrix", &core::get_source_matrix)
		.def("get_target_matrix", &core::get_target_matrix);
}
