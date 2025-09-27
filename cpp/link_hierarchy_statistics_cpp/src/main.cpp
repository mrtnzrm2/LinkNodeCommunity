

/*
 * link_hierarchy_statistics_cpp/src/main.cpp
 *
 * Library: link_hierarchy_statistics_cpp
 * Author: Jorge S. Martinez Armas
 *
 * Overview:
 * ----------
 * This library provides efficient computation of link community statistics across the link community hierarchy
 * of a graph. It is designed to help uncover hierarchical resolutions with optimal properties, such as density
 * and entropy, by incrementally tracking statistics as edges are merged during hierarchical clustering.
 *
 * Main Components:
 * ----------------
 * - core class: The central class that manages the computation of link community statistics.
 *   It supports both matrix-based and edge-list-based hierarchical clustering, and exposes
 *   methods for fitting the model and retrieving computed statistics.
 *
 * - LinkCommunityStats struct: Stores statistics for each link community, including node membership,
 *   number of nodes/edges, excess edges, density, and entropy.
 *
 * - Utility functions: Compute density (Dc), excess-link (loop) entropy (Sc), and tree probability (Ptree)
 *   for link communities, with robust error checking.
 *
 * core Class Parameters:
 * ----------------------
 * - N (int): Number of nodes in the analyzed graph.
 * - M (int): Number of edges in the analyzed graph.
 * - source_nodes (std::vector<int>): List of source node indices for each edge.
 * - target_nodes (std::vector<int>): List of target node indices for each edge.
 * - linkage (int): Linkage method for hierarchical clustering (single linkage recommended).
 * - undirected (bool): Whether the graph is undirected.
 * - edge_complete (bool): Indicates if the provided edges already correspond to the edge-complete graph.
 *   When false, inputs are filtered to retain nodes present as both sources and targets (< N).
 * - verbose (int, optional): Verbosity level (0 no output, 1 workflow milestones, 2 step annoucements).
 * - force (bool, optional): Ignore recoverable runtime errors to let workflows finish when possible.
 *   False (0, default) halts on errors, True (1) attempts to continue.
 *
 * core Class Methods:
 * -------------------
 * - fit_matrix(std::vector<double>& condensed_distance_matrix):
 *     Fits the hierarchical clustering model using a condensed distance matrix (upper triangle).
 *     Computes link community statistics incrementally as clusters are merged.
 *
 * - fit_edgelist(std::vector<std::vector<double>>& distance_edgelist, const double& max_dist):
 *     Fits the hierarchical clustering model using an edge list with distances.
 *     Computes link community statistics incrementally as clusters are merged.
 *
 * - get_K(): Returns the number of link communities at each step of the hierarchy.
 * - get_Height(): Returns the merge heights at each step.
 * - get_D(): Returns the incremental link community density at each step.
 * - get_S(): Returns the incremental excess-link (loop) entropy at each step.
 *
 * Implementation Notes:
 * ---------------------
 * - The library is robust to malformed input and out-of-bounds indices.
 * - When edge_complete is false, the class preprocesses source/target inputs to operate on the induced subgraph.
 * - Self-loops are not supported and will halt execution.
 * - Statistics are updated incrementally for merged clusters only, improving performance.
 * - The linkage matrix must follow the R/fastcluster convention (negative for singletons, positive for compounds).
 *
 * Usage:
 * ------
 * Instantiate the core class with graph parameters, fit the model using either fit_matrix or fit_edgelist,
 * and retrieve statistics using the provided getter methods. The library is exposed to Python via pybind11.
 */
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <utility>
#include "../../libs/hclust-cpp/fastcluster.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;


double Dc(int &m, int &n, bool &undirected) {
  double dc;
  if (!undirected) {
    if (n < 2) {
      throw std::invalid_argument("Dc (directed) requires n >= 2");
    }
    double denom = pow(n - 1., 2.);
    if (denom == 0.0) {
      throw std::runtime_error("Dc denominator zero (directed); returning 0.");
    }
    dc = (m - n + 1.) / denom;
  }
  else {
    if (n == 2 && m == 1) return 0; // Special case: two nodes, one edge
    if (n < 3) {
      throw std::invalid_argument("Dc (undirected) requires n >= 3");
    }
    double denom = ((n * (n - 1.) / 2.) - n + 1.);
    if (denom == 0.0) {
      throw std::runtime_error("Dc denominator zero (undirected); returning 0.");
    }
    dc = (m - n + 1.) / denom;
  }
  if (dc >= 0) return dc;
  else {
    throw std::runtime_error("Link community density (Dc) was negative.");
  }
}

double Sc(int &m, int &n, int &M, int& N) {
  double pc;
  if ((M - N + 1) == 0) {
    throw std::runtime_error("Denominator zero in Sc; returning 0.");
  }
  pc = (m - n + 1.) / (M - N + 1.);
  if (pc > 0) return -pc * log(pc);
  else if (pc == 0) return 0;
  else {
    throw std::runtime_error("Excess link entropy (Sc) was negative.");
  }
}

double Ptree(int &M, int& N, int& mtree) {
  double ptree;
  if ((M - N + 1) == 0) {
    throw std::runtime_error("Denominator zero in Ptree; returning 0.");
  }
  ptree = 1 - (mtree / (M  - N + 1.));
  if (ptree > 0) return -ptree * log(ptree);
  else if (ptree == 0) return 0;
  else {
    throw std::runtime_error("Probability of being a tree (Ptree) was negative.");
  }
}

// Transform std::vector<std::vector<double>> to std::vector<edge_struct>
std::vector<edge_struct> vecvec_to_edgelist(const std::vector<std::vector<double>>& edgelist_vecvec) {
    std::vector<edge_struct> edgelist;
    edgelist.reserve(edgelist_vecvec.size());
    for (const auto& v : edgelist_vecvec) {
        if (v.size() != 3) continue; // Only accept triples
        edgelist.emplace_back(
            static_cast<int>(v[0]),
            static_cast<int>(v[1]),
            v[2]
        );
    }
    return edgelist;
}

struct LinkCommunityStats {
    std::set<int> node_members;
    int n; // number of nodes
    int m; // number of edges
    int mtree; // number of excess edges from the minimum spanning tree
    double dc; // link community density
    double sc; // excess-link probability
};

// Core class for link hierarchy statistics computation.
// Manages clustering, statistics calculation, and exposes workflow methods.
class core {
  private:
    int number_of_nodes;
    int number_of_edges;
    int linkage;
    bool undirected;
    int verbose;
    bool force;
    bool edge_complete;

    std::vector<int> source_nodes;
    std::vector<int> target_nodes;

    // Output vectors for statistics at each hierarchy step
    std::vector<int> K;           // Number of link communities
    std::vector<double> Height;   // Merge heights
    std::vector<double> D;        // Link community density
    std::vector<double> S;        // Excess-link entropy

    std::pair<std::vector<int>, std::vector<int>> filter_and_sort_edges(
        const std::vector<int>& source,
        const std::vector<int>& target
    ) const;

  public:
    // Constructor: initializes graph and clustering parameters
    core(
      const int N,
      const int M,
      std::vector<int> source_nodes,
      std::vector<int> target_nodes,
      const int linkage,
      const bool undirected,
      const int verbose = 0,
      const bool force = false,
      const bool edge_complete = true
    );
    ~core(){};

    // Fit hierarchical clustering using condensed distance matrix (upper triangle)
    // Computes link community statistics incrementally as clusters are merged.
    void fit_matrix(std::vector<double> &condensed_distance_matrix);

    // Fit hierarchical clustering using edge list with distances
    // Computes link community statistics incrementally as clusters are merged.
    void fit_edgelist(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist);

    // Utility: expand vector to required size and initialize to zero
    template <typename T>
    void expand_vector(std::vector<T>& v, const int& N);

    // Getters for computed statistics (workflow: retrieve after fitting)
    std::vector<int> get_K();         // Number of link communities at each step
    std::vector<double> get_Height(); // Merge heights at each step
    std::vector<double> get_D();      // Incremental link community density
    std::vector<double> get_S();      // Incremental excess-link entropy
};


core::core(
  const int N, // number of nodes in the graph analyzed
  const int M, // number of edges in the graph analyzed
  std::vector<int> source_nodes,
  std::vector<int> target_nodes,
  const int linkage,
  const bool undirected,
  const int verbose,
  const bool force,
  const bool edge_complete
) {
  number_of_nodes = N;
  number_of_edges = M;
  this->source_nodes = source_nodes;
  this->target_nodes = target_nodes;
  this->linkage = linkage;
  this->undirected = undirected;
  this->force = force;
  this->edge_complete = edge_complete;

  if (verbose < 0 || verbose > 2) {
    std::cerr << "Warning: Verbose level " << verbose << " out of range [0,2]. Clamping to nearest bound.\n";
    this->verbose = std::max(0, std::min(verbose, 2));
  } else {
    this->verbose = verbose;
  }

  if (this->linkage != HCLUST_METHOD_SINGLE) {
    std::cerr << "Warning: Non-single linkage selected (" << this->linkage << "). For link community analysis, single linkage is recommended. Other methods may produce artifactual results.\n";
  }

  if (this->verbose >= 1) {
    std::cout << "[linkstat] Initialized with N=" << number_of_nodes
              << ", M=" << number_of_edges
              << ", verbose=" << this->verbose
              << ", force=" << (this->force ? 1 : 0)
              << ", edge_complete=" << (this->edge_complete ? 1 : 0) << "\n";
  }
}

std::pair<std::vector<int>, std::vector<int>> core::filter_and_sort_edges(
    const std::vector<int>& source,
    const std::vector<int>& target
) const {
  if (source.size() != target.size()) {
    throw std::invalid_argument("Source and target vectors must have the same size.");
  }

  std::vector<std::pair<int, int>> filtered_edges;
  filtered_edges.reserve(source.size());

  for (size_t i = 0; i < source.size(); ++i) {
    const int s = source[i];
    const int t = target[i];

    if (s < 0 || t < 0) {
      throw std::out_of_range("Source/target indices must be non-negative.");
    }

    if (s < number_of_nodes && t < number_of_nodes) {
      filtered_edges.emplace_back(s, t);
    }
  }

  std::stable_sort(
    filtered_edges.begin(),
    filtered_edges.end(),
    [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
      if (lhs.first != rhs.first) return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    }
  );

  std::vector<int> sorted_source;
  std::vector<int> sorted_target;
  sorted_source.reserve(filtered_edges.size());
  sorted_target.reserve(filtered_edges.size());

  for (const auto& edge : filtered_edges) {
    sorted_source.push_back(edge.first);
    sorted_target.push_back(edge.second);
  }

  return {std::move(sorted_source), std::move(sorted_target)};
}

// Optimized fit method for ph class -- enhanced logic, variable naming, and documentation.
// This implementation incrementally computes link community statistics using the linkage matrix produced by hierarchical clustering.
// The code avoids recomputing statistics for all communities at every step, updating only the merged clusters, and maintains robust error checking.
void core::fit_edgelist(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist) {
    try {
        // --- Variable Declarations ---
        // Statistics and loop variables
        int prev_tree_excess, tree_excess_sum;

        // linkage matrix, cluster mapping
        std::vector<std::vector<double>> linkage_matrix;

        // Object mapping link communities to their statistics
        std::unordered_map<int, LinkCommunityStats> cluster_map;

        if (verbose >= 1) {
            std::cout << "[linkstat] Starting fit_edgelist workflow" << std::endl;
        }
        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 1: validating inputs" << std::endl;
        }

        // --- Input Validation ---
        if (source_nodes.size() != target_nodes.size()) {
            throw std::invalid_argument("Source and target vectors must have the same size.");
        }

        std::vector<int> processed_source_nodes;
        std::vector<int> processed_target_nodes;
        if (!edge_complete) {
            if (verbose >= 2) {
                std::cout << "[linkstat] fit_edgelist Step 1a: filtering and sorting edge list" << std::endl;
            }
            auto filtered_edges = filter_and_sort_edges(source_nodes, target_nodes);
            processed_source_nodes = std::move(filtered_edges.first);
            processed_target_nodes = std::move(filtered_edges.second);
        }

        const std::vector<int>& current_source_nodes = edge_complete ? source_nodes : processed_source_nodes;
        const std::vector<int>& current_target_nodes = edge_complete ? target_nodes : processed_target_nodes;

        if (current_source_nodes.size() != static_cast<size_t>(number_of_edges) ||
            current_target_nodes.size() != static_cast<size_t>(number_of_edges)) {
            throw std::invalid_argument("Processed source/target vectors must match number_of_edges.");
        }

        for (size_t i = 0; i < current_source_nodes.size(); ++i) {
            if (current_source_nodes[i] < 0 || current_source_nodes[i] >= number_of_nodes ||
                current_target_nodes[i] < 0 || current_target_nodes[i] >= number_of_nodes) {
                throw std::out_of_range("Source/target indices out of bounds.");
            }
        }

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 2: preparing clustering data" << std::endl;
        }
        // --- Prepare Clustering ---
        std::vector<edge_struct> edgelist = vecvec_to_edgelist(distance_edgelist);

        // Allocate clustering results
        std::vector<int> merge(2 * (number_of_edges - 1));
        std::vector<double> height(number_of_edges - 1);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 3: running hierarchical clustering" << std::endl;
        }
        // --- Run Clustering ---
        hclust_fast_edgelist(number_of_edges, edgelist, max_dist, merge.data(), height.data());
        linkage_matrix = make_linkage_matrix(merge.data(), height.data(), number_of_edges);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 4: preparing output buffers" << std::endl;
        }
        // --- Prepare Output Vectors ---
        expand_vector(K, number_of_edges - 1);
        expand_vector(Height, number_of_edges - 1);
        expand_vector(D, number_of_edges - 1);
        expand_vector(S, number_of_edges - 1);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 5: initializing cluster map" << std::endl;
        }
        // --- Initialize Link Community Map ---
        // Each initial edge (singleton cluster) is mapped to its nodes and statistics
        tree_excess_sum = 0;
        for (int i = 0; i < number_of_edges; ++i) {
            LinkCommunityStats lcn;

            lcn.node_members.insert(current_source_nodes[i]);
            lcn.node_members.insert(current_target_nodes[i]);
            if (current_source_nodes[i] != current_target_nodes[i]) {
                lcn.n = 2;
            } else {
                // Self-loops are not allowed, halt execution
                throw std::invalid_argument("Self-loops detected. Self-loops are not allowed in this implementation.");
            }
            lcn.m = 1;
            lcn.dc = Dc(lcn.m, lcn.n, undirected);
            lcn.sc = Sc(lcn.m, lcn.n, number_of_edges, number_of_nodes);
            lcn.mtree = lcn.m - lcn.n + 1;
            cluster_map[-i - 1] = lcn;

            tree_excess_sum += lcn.mtree;
        }
        prev_tree_excess = tree_excess_sum;


        double running_density = 0, running_entropy = 0;

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_edgelist Step 6: processing linkage merges" << std::endl;
        }
        // --- Main Incremental Hierarchy Loop ---
        for (size_t i = 0; i < linkage_matrix.size(); i++) {
            // 1. Determine number of clusters at this step
            int num_link_communities = number_of_edges - static_cast<int>(i) - 1;

            // 2. Retrieve indices of clusters being merged (R convention)
            int idx1 = static_cast<int>(linkage_matrix[i][0]);
            int idx2 = static_cast<int>(linkage_matrix[i][1]);
            double merge_height = linkage_matrix[i][2]; // Height of current merge

            K[i] = num_link_communities;
            Height[i] = merge_height;

            // 3. Merge cluster properties
            LinkCommunityStats merged_lcn;
            // Validate indices exist in map to avoid accidental default insertion
            const auto &c1 = cluster_map.at(idx1);
            const auto &c2 = cluster_map.at(idx2);
            // Merge node sets
            merged_lcn.node_members = c1.node_members;
            merged_lcn.node_members.insert(
                c2.node_members.begin(),
                c2.node_members.end()
            );
            merged_lcn.n = merged_lcn.node_members.size();
            merged_lcn.m = c1.m + c2.m;
            merged_lcn.mtree = merged_lcn.m - merged_lcn.n + 1;

            try {
                // Update community density and entropy
                merged_lcn.dc = Dc(merged_lcn.m, merged_lcn.n, undirected);
                merged_lcn.sc = Sc(merged_lcn.m, merged_lcn.n, number_of_edges, number_of_nodes);

                // 4. Update cluster map with new merged cluster
                cluster_map[static_cast<int>(i) + 1] = merged_lcn;

                // 5. Update excess tree edges ("mtree") for normalization
                tree_excess_sum += merged_lcn.mtree - (c1.mtree + c2.mtree);

                // 6. Incremental update of statistics
                running_density += merged_lcn.dc * merged_lcn.m / number_of_edges
                    - (c1.dc * c1.m / number_of_edges + c2.dc * c2.m / number_of_edges);
                running_entropy += merged_lcn.sc - (c1.sc + c2.sc);
                running_entropy += Ptree(number_of_edges, number_of_nodes, tree_excess_sum)
                    - Ptree(number_of_edges, number_of_nodes, prev_tree_excess);
                prev_tree_excess = tree_excess_sum;

                D[i] = running_density;
                S[i] = running_entropy;
            } catch (const std::exception& ex) {
                std::cerr << "[linkstat] fit_edgelist merge failure at step=" << i
                          << " merge_indices=(" << idx1 << ", " << idx2 << ")"
                          << " merged_stats(n=" << merged_lcn.n
                          << ", m=" << merged_lcn.m
                          << ", mtree=" << merged_lcn.mtree << ")"
                          << " error=" << ex.what() << std::endl;
                throw;
            }

            // 7. Cleanup merged clusters from map to save memory
            cluster_map.erase(idx1);
            cluster_map.erase(idx2);

            // --- Implementation Notes ---
            // - All statistics are updated incrementally based on the linkage matrix.
            // - Only clusters involved in each merge need updating; all others are unchanged.
            // - The linkage matrix is assumed to follow R/fastcluster convention: singletons negative, compounds positive.
            // - If your implementation changes this convention, update idx1/idx2 usage accordingly.
        }

        if (verbose >= 1) {
            std::cout << "[linkstat] Completed fit_edgelist workflow" << std::endl;
        }
    } catch (const std::exception& ex) {
        if (!force) {
            throw;
        }
        if (verbose >= 1) {
            std::cerr << "[linkstat] fit_edgelist forced completion after error: "
                      << ex.what() << std::endl;
        }
    }
}

// Optimized fit method for ph class -- enhanced logic, variable naming, and documentation.
// This implementation incrementally computes link community statistics using the linkage matrix produced by hierarchical clustering.
// The code avoids recomputing statistics for all communities at every step, updating only the merged clusters, and maintains robust error checking.

void core::fit_matrix(std::vector<double>& condensed_distance_matrix) {
    try {
        // --- Variable Declarations ---
        // Statistics and loop variables
        int prev_tree_excess, tree_excess_sum;
        long int size_condensed_matrix = (static_cast<long int>(number_of_edges) * static_cast<long int>(number_of_edges - 1)) / 2;

        // linkage matrix, cluster mapping
        std::vector<std::vector<double>> linkage_matrix;

        // Object map for link communities to their statistics
        std::unordered_map<int, LinkCommunityStats> cluster_map;

        if (verbose >= 1) {
            std::cout << "[linkstat] Starting fit_matrix workflow" << std::endl;
        }
        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 1: validating inputs" << std::endl;
        }

        // --- Input Validation ---
        if (source_nodes.size() != target_nodes.size()) {
            throw std::invalid_argument("Source and target vectors must have the same size.");
        }

        std::vector<int> processed_source_nodes;
        std::vector<int> processed_target_nodes;
        if (!edge_complete) {
            if (verbose >= 2) {
                std::cout << "[linkstat] fit_matrix Step 1a: filtering and sorting edge list" << std::endl;
            }
            auto filtered_edges = filter_and_sort_edges(source_nodes, target_nodes);
            processed_source_nodes = std::move(filtered_edges.first);
            processed_target_nodes = std::move(filtered_edges.second);
        }

        const std::vector<int>& current_source_nodes = edge_complete ? source_nodes : processed_source_nodes;
        const std::vector<int>& current_target_nodes = edge_complete ? target_nodes : processed_target_nodes;

        if (current_source_nodes.size() != static_cast<size_t>(number_of_edges) ||
            current_target_nodes.size() != static_cast<size_t>(number_of_edges)) {
            throw std::invalid_argument("Processed source/target vectors must match number_of_edges.");
        }

        if (condensed_distance_matrix.size() != size_condensed_matrix) {
            throw std::invalid_argument("Distance vector must be a condensed upper triangle matrix.");
        }

        for (size_t i = 0; i < current_source_nodes.size(); ++i) {
            if (current_source_nodes[i] < 0 || current_source_nodes[i] >= number_of_nodes ||
                current_target_nodes[i] < 0 || current_target_nodes[i] >= number_of_nodes) {
                throw std::out_of_range("Source/target indices out of bounds.");
            }
        }

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 2: preparing condensed distance array" << std::endl;
        }
        // --- Prepare Clustering ---
        // Allocate and fill condensed distance matrix
        std::vector<double> condensed_distance_array = condensed_distance_matrix;

        // Allocate clustering results
        std::vector<int> merge(2 * (number_of_edges - 1));
        std::vector<double> height(number_of_edges - 1);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 3: running hierarchical clustering" << std::endl;
        }
        // --- Run Clustering ---
        hclust_fast(number_of_edges, condensed_distance_array.data(), linkage, merge.data(), height.data());
        linkage_matrix = make_linkage_matrix(merge.data(), height.data(), number_of_edges);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 4: preparing output buffers" << std::endl;
        }
        // --- Prepare Output Vectors ---
        expand_vector(K, number_of_edges - 1);
        expand_vector(Height, number_of_edges - 1);
        expand_vector(D, number_of_edges - 1);
        expand_vector(S, number_of_edges - 1);

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 5: initializing cluster map" << std::endl;
        }
        // --- Initialize Link Community Map ---
        // Each initial edge (singleton cluster) is mapped to its nodes and statistics
        tree_excess_sum = 0;
        for (int i = 0; i < number_of_edges; ++i) {
            LinkCommunityStats lcn;
            lcn.node_members.insert(current_source_nodes[i]);
            lcn.node_members.insert(current_target_nodes[i]);
            if (current_source_nodes[i] != current_target_nodes[i]) {
                lcn.n = 2;
            } else {
                // Self-loops are not allowed, halt execution
                throw std::invalid_argument("Self-loops detected. Self-loops are not allowed in this implementation.");
            }
            lcn.m = 1;
            lcn.dc = Dc(lcn.m, lcn.n, undirected);
            lcn.sc = Sc(lcn.m, lcn.n, number_of_edges, number_of_nodes);
            lcn.mtree = lcn.m - lcn.n + 1;
            cluster_map[-i - 1] = lcn;

            tree_excess_sum += lcn.mtree;
        }
        prev_tree_excess = tree_excess_sum;

        double running_density = 0, running_entropy = 0;

        if (verbose >= 2) {
            std::cout << "[linkstat] fit_matrix Step 6: processing linkage merges" << std::endl;
        }
        // --- Main Incremental Hierarchy Loop ---
        for (size_t i = 0; i < linkage_matrix.size(); i++) {
            // 1. Determine number of clusters at this step
            int num_link_communities = number_of_edges - static_cast<int>(i) - 1;

            // 2. Retrieve indices of clusters being merged (R convention)
            int idx1 = static_cast<int>(linkage_matrix[i][0]);
            int idx2 = static_cast<int>(linkage_matrix[i][1]);
            double merge_height = linkage_matrix[i][2]; // Height of current merge

            K[i] = num_link_communities;
            Height[i] = merge_height;

            // 3. Merge cluster properties
            LinkCommunityStats merged_lcn;
            // Validate indices exist in map to avoid accidental default insertion
            const auto &c1 = cluster_map.at(idx1);
            const auto &c2 = cluster_map.at(idx2);
            // Merge node sets
            merged_lcn.node_members = c1.node_members;
            merged_lcn.node_members.insert(
                c2.node_members.begin(),
                c2.node_members.end()
            );
            merged_lcn.n = merged_lcn.node_members.size();
            merged_lcn.m = c1.m + c2.m;
            merged_lcn.mtree = merged_lcn.m - merged_lcn.n + 1;

            try {
                // Update community density and entropy
                merged_lcn.dc = Dc(merged_lcn.m, merged_lcn.n, undirected);
                merged_lcn.sc = Sc(merged_lcn.m, merged_lcn.n, number_of_edges, number_of_nodes);

                // 4. Update cluster map with new merged cluster
                cluster_map[static_cast<int>(i) + 1] = merged_lcn;

                // 5. Update excess tree edges ("mtree") for normalization
                tree_excess_sum += merged_lcn.mtree - (c1.mtree + c2.mtree);

                // 6. Incremental update of statistics
                running_density += merged_lcn.dc * merged_lcn.m / number_of_edges
                    - (c1.dc * c1.m / number_of_edges + c2.dc * c2.m / number_of_edges);
                running_entropy += merged_lcn.sc - (c1.sc + c2.sc);
                running_entropy += Ptree(number_of_edges, number_of_nodes, tree_excess_sum)
                    - Ptree(number_of_edges, number_of_nodes, prev_tree_excess);
                prev_tree_excess = tree_excess_sum;

                D[i] = running_density;
                S[i] = running_entropy;
            } catch (const std::exception& ex) {
                std::cerr << "[linkstat] fit_matrix merge failure at step=" << i
                          << " merge_indices=(" << idx1 << ", " << idx2 << ")"
                          << " merged_stats(n=" << merged_lcn.n
                          << ", m=" << merged_lcn.m
                          << ", mtree=" << merged_lcn.mtree << ")"
                          << " error=" << ex.what() << std::endl;
                throw;
            }

            // 7. Cleanup merged clusters from map to save memory
            cluster_map.erase(idx1);
            cluster_map.erase(idx2);

            // --- Implementation Notes ---
            // - All statistics are updated incrementally based on the linkage matrix.
            // - Only clusters involved in each merge need updating; all others are unchanged.
            // - The linkage matrix is assumed to follow R/fastcluster convention: singletons negative, compounds positive.
            // - If your implementation changes this convention, update idx1/idx2 usage accordingly.
        }

        if (verbose >= 1) {
            std::cout << "[linkstat] Completed fit_matrix workflow" << std::endl;
        }
    } catch (const std::exception& ex) {
        if (!force) {
            throw;
        }
        if (verbose >= 1) {
            std::cerr << "[linkstat] fit_matrix forced completion after error: "
                      << ex.what() << std::endl;
        }
    }
}

// --- Implementation Caveats ---
// - Self-loops are not supported; the code will halt if detected in the input.
// - The linkage matrix must be produced in R convention (see fastcluster/make_linkage_matrix).
// - Output vectors K, Height, D, S are preallocated to size N-1 (number_of_edges - 1) matching the clustering steps.
// - The function is robust to out-of-bounds and malformed input.
// - Statistics are only updated for merged clusters, which greatly improves performance over naive implementations.

template <typename T>
void core::expand_vector(std::vector<T>& v, const int& N) {v = std::vector<T>(N, 0);}

std::vector<int> core::get_K() {return K;}
std::vector<double> core::get_Height() {return Height;}
std::vector<double> core::get_D() {return D;}
std::vector<double> core::get_S() {return S;}


PYBIND11_MODULE(link_hierarchy_statistics_cpp, m) {
    py::class_<core>(m, "core", py::module_local())
        .def(
          py::init<
            const int,
            const int,
            std::vector<int>,
            std::vector<int>,
            const int,
            const bool,
            const int,
            const bool,
            const bool
          >(),
          py::arg("N"),
          py::arg("M"),
          py::arg("source_nodes"),
          py::arg("target_nodes"),
          py::arg("linkage"),
          py::arg("undirected"),
          py::arg("verbose") = 0,
          py::arg("force") = false,
          py::arg("edge_complete") = true
        )

        .def("fit_matrix", &core::fit_matrix)
        .def("fit_edgelist", &core::fit_edgelist)
        .def("get_K", &core::get_K)
        .def("get_Height", &core::get_Height)
        .def("get_D", &core::get_D)
        .def("get_S", &core::get_S);
}
