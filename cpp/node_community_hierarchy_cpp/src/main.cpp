/*
 * cpp/node_community_hierarchy_cpp/src/main.cpp
 *
 * Library: node_community_hierarchy_cpp
 * Author: Jorge S. Martinez Armas
 *
 * Overview:
 * ----------
 * This library derives a node–community hierarchy from a link–community hierarchy.
 * Given a hierarchical clustering of links (built either from a condensed distance
 * matrix or a sparse edge list), it incrementally merges node communities according
 * on specific node–merging criteria defined in this file. The result is a dendrogram
 * over nodes that reveals the hierarchical community structure of the network.
 * Both directed graphs and undirected graphs (experimental; slower) are supported.
 *
 * Main Components:
 * ----------------
 * - core class: Drives the overall workflow. It builds the link hierarchy using
 *   fastcluster, tracks the evolving membership of link communities, detects
 *   node overlaps/cliques that satisfy the merge criteria, and records node
 *   merge events (node dendrogram) and equivalence tracking across steps.
 *
 * - merge_node_communities(): Encapsulates the node‑merge logic. When the link
 *   hierarchy indicates two (or more) compatible link communities, their induced
 *   node sets are intersected to detect cliques/overlaps. If the intersection is
 *   non‑trivial, corresponding node communities are merged and the node hierarchy
 *   is updated. This function holds the core merging criteria of the approach.
 *
 * - NeighborNodes / LinkCommunityNodes: Lightweight containers used to maintain
 *   adjacency information for undirected mode (NeighborNodes) and to track source
 *   and target node sets for directed mode (LinkCommunityNodes).
 *
 * core Class Parameters:
 * ----------------------
 * - N (int): Number of nodes in the analyzed graph.
 * - M (int): Number of edges (links) in the analyzed graph.
 * - source_nodes (std::vector<int>): Source index for each edge (size M).
 * - target_nodes (std::vector<int>): Target index for each edge (size M).
 * - linkage (int): Linkage method for the link hierarchy (single recommended).
 * - undirected (int/bool): 0 for directed (default), 1 for undirected (experimental).
 *
 * core Class Methods:
 * -------------------
 * - fit_matrix(std::vector<double>& cdm): Build link hierarchy from condensed distance
 *   matrix (size M*(M-1)/2) and derive the node hierarchy via the merge criteria.
 * - fit_edgelist(std::vector<std::vector<double>>& el, const double& max_dist): Build
 *   link hierarchy from a sparse edge list of link‑pairs with distances and derive
 *   the node hierarchy. max_dist is the padding distance for disconnected components.
 * - get_node_hierarchy(): Return the node dendrogram as (N-1) x 4 linkage rows
 *   [merge1, merge2, height, cluster_size] in R/fastcluster convention.
 * - get_linknode_equivalence(): Return per‑step equivalence tracking between link and
 *   node communities (for analysis/visualization of co‑evolution).
 *
 * Implementation Notes:
 * ---------------------
 * - The link hierarchy uses the fastcluster backend (single linkage recommended).
 * - Node‑merge criteria are applied incrementally at each link‑merge step and are
 *   the key to uncovering meaningful node communities.
 * - The undirected mode uses NeighborNodes and a triangle‑based intersection routine
 *   to find nodes participating in triads; the directed mode uses intersections of
 *   source/target sets induced by merged link communities.
 * - Inputs are validated; indices are expected to be 0‑based for nodes and 1..M for
 *   link identifiers in the fastcluster outputs.
 *
 * Usage:
 * ------
 * Instantiate core with (N, M, source_nodes, target_nodes, linkage, undirected),
 * call fit_matrix() or fit_edgelist(), and query results with get_node_hierarchy()
 * and get_linknode_equivalence(). The module is exposed to Python via pybind11.
 */

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <unordered_set>
#include <iterator>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../libs/hclust-cpp/fastcluster.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals; 
namespace py = pybind11;

// Forward declaration for edge-list clustering (implemented in fastcluster.cpp)
int hclust_fast_edgelist(int n, const std::vector<edge_struct>& edges, double max_dist, int* merge, double* height);

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

void unique(std::vector<int>& v) {
  std::vector<int>::iterator ip;
  std::sort(v.begin(), v.end());
  ip = std::unique(
    v.begin(),
    v.begin() + v.size()
  );
  v.resize(
    std::distance(v.begin(), ip)
  );
}

std::vector<int> intersect(
  const std::vector<int>& v,
  const std::vector<int>& u
) {
  std::vector<int> intersect;
  std::set_intersection(
    v.begin(), v.end(),
    u.begin(), u.end(),
    std::back_inserter(intersect)
  );
  return intersect;
}

struct node {
  int merge;
  std::vector<int> members;
};

struct LinkCommunityNodes {
    std::vector<int> source;
    std::vector<int> target;
};

// Merges node communities based on intersection/clique and updates hierarchy/equivalence
void merge_node_communities(
  std::vector<node>& merge_list,
  const std::vector<int>& intersection_nodes,
  int& merge_step_counter,
  int k,
  double merge_height,
  int nodes,
  std::vector<int>& eq_pair,
  bool& eq_true,
  std::vector<std::vector<double>>& node_hierarchy,
  std::vector<std::vector<int>>& linknode_equivalence
) {
  if (intersection_nodes.size() > 1) {
      std::vector<node> merge_list_clone = merge_list;
      std::vector<int> compatible_communities;
      std::vector<int> new_members;
      node new_node;
      for (size_t i = 0; i < merge_list_clone.size(); ++i) {
          std::vector<int> inter = intersect(intersection_nodes, merge_list_clone[i].members);
          if (!inter.empty()) compatible_communities.push_back(static_cast<int>(i));
      }
      // Merge pairs of compatible communities
      if (compatible_communities.size() == 2) {
          eq_true = true;
          node_hierarchy[merge_step_counter][0] = merge_list_clone[compatible_communities[0]].merge;
          node_hierarchy[merge_step_counter][1] = merge_list_clone[compatible_communities[1]].merge;
          node_hierarchy[merge_step_counter][2] = merge_height;
          node_hierarchy[merge_step_counter][3] =
              merge_list_clone[compatible_communities[0]].members.size() +
              merge_list_clone[compatible_communities[1]].members.size();
          new_members = merge_list_clone[compatible_communities[0]].members;
          new_members.insert(new_members.end(),
              merge_list_clone[compatible_communities[1]].members.begin(),
              merge_list_clone[compatible_communities[1]].members.end());
          unique(new_members);
          new_node = { merge_step_counter + nodes, new_members };
          for (int rm = 0; rm < 2; ++rm) {
              merge_list_clone.erase(merge_list_clone.begin() + compatible_communities[rm] - rm);
          }
          merge_list_clone.push_back(new_node);
          eq_pair[0] = k;
          eq_pair[1] = nodes - merge_step_counter - 1;
          linknode_equivalence.push_back(eq_pair);
          merge_step_counter++;
      }
      // Merge more than 2 compatible communities
      else if (compatible_communities.size() > 2) {
          eq_true = true;
          std::vector<node> merge_list_clone_2;
          for (int idx : compatible_communities) {
              merge_list_clone_2.push_back(merge_list_clone[idx]);
          }
          for (size_t rm = 0; rm < compatible_communities.size(); ++rm) {
              merge_list_clone.erase(merge_list_clone.begin() + compatible_communities[rm] - rm);
          }
          while (merge_list_clone_2.size() >= 2) {
              node_hierarchy[merge_step_counter][0] = merge_list_clone_2[0].merge;
              node_hierarchy[merge_step_counter][1] = merge_list_clone_2[1].merge;
              node_hierarchy[merge_step_counter][2] = merge_height;
              node_hierarchy[merge_step_counter][3] = merge_list_clone_2[0].members.size() + merge_list_clone_2[1].members.size();
              std::vector<int> new_members = merge_list_clone_2[0].members;
              new_members.insert(
                  new_members.end(),
                  merge_list_clone_2[1].members.begin(),
                  merge_list_clone_2[1].members.end()
              );
              unique(new_members);
              node new_node = { merge_step_counter + nodes, new_members };
              merge_list_clone_2.erase(merge_list_clone_2.begin(), merge_list_clone_2.begin() + 2);
              if (merge_list_clone_2.empty()) {
                  merge_list_clone.push_back(new_node);
              } else {
                  merge_list_clone_2.push_back(new_node);
              }
              eq_pair[0] = k;
              eq_pair[1] = nodes - merge_step_counter - 1;
              linknode_equivalence.push_back(eq_pair);
              merge_step_counter++;
          }
      }
      merge_list = merge_list_clone;
    }
}

// Class to manage adjacency lists and node degrees for undirected graphs
class NeighborNodes {
  public:
    // Adjacency list: node -> set of neighbors (no duplicates)
    std::unordered_map<int, std::set<int>> adjacency_list;
    // Node degrees: node -> degree
    std::unordered_map<int, int> degrees;

    // Default constructor: initialize empty adjacency list and degrees
    NeighborNodes() {}

    // Constructor: initialize with a pair of node indices
    NeighborNodes(int node1, int node2) {
      add_edge(node1, node2);
    }

    // Constructor: initialize with two adjacency lists
    NeighborNodes(const std::unordered_map<int, std::set<int>>& adj1,
                  const std::unordered_map<int, std::set<int>>& adj2) {
      combine(adj1, adj2);
    }

    // Add an undirected edge between node1 and node2
    void add_edge(int node1, int node2) {
      adjacency_list[node1].insert(node2);
      adjacency_list[node2].insert(node1);
      degrees[node1] = adjacency_list[node1].size();
      degrees[node2] = adjacency_list[node2].size();
    }

    // Combine two adjacency lists and update degrees
    void combine(const std::unordered_map<int, std::set<int>>& adj1,
                 const std::unordered_map<int, std::set<int>>& adj2) {
      // Merge adj1
      for (const auto& kv : adj1) {
        adjacency_list[kv.first].insert(kv.second.begin(), kv.second.end());
      }
      // Merge adj2
      for (const auto& kv : adj2) {
        adjacency_list[kv.first].insert(kv.second.begin(), kv.second.end());
      }
      // Update degrees
      for (auto& kv : adjacency_list) {
        degrees[kv.first] = kv.second.size();
      }
    }
};

/**
 * Computes the number of nodes that belong to a triad (3-clique) using a degree-oriented neighbor intersection algorithm.
 * @param neighbors - NeighborNodes instance containing adjacency list and degrees.
 * @return Nodes that are part of at least one triad (3-clique).
 */
std::vector<int> nodes_in_triads(const NeighborNodes& G) {
  // Helper: rank = (degree, node_id), ascending; ties broken by node_id.
  auto rank_of = [&](int x) -> std::pair<int,int> {
    auto it = G.degrees.find(x);
    int d = (it == G.degrees.end()) ? 0 : it->second;
    return {d, x};
  };
  auto rank_less = [&](int a, int b) {
    auto ra = rank_of(a), rb = rank_of(b);
    if (ra.first != rb.first) return ra.first < rb.first;
    return ra.second < rb.second;
  };

  // Build forward neighbor lists: keep only neighbors with higher rank.
  std::unordered_map<int, std::vector<int>> fwd;
  fwd.reserve(G.adjacency_list.size());

  for (const auto& kv : G.adjacency_list) {
    int u = kv.first;
    const auto& nbrs = kv.second; // std::set<int>, sorted by node_id
    auto& out = fwd[u];
    out.reserve(nbrs.size());
    for (int v : nbrs) {
      if (u == v) continue;              // ignore self-loops if any
      if (rank_less(u, v)) out.push_back(v);
    }
    // Sort by rank so we can do two-pointer intersections efficiently.
    std::sort(out.begin(), out.end(), rank_less);
  }

  // Parallelized version using OpenMP and thread-local sets
int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
std::vector<std::unordered_set<int>> thread_triangles(num_threads);

// Collect keys for deterministic and thread-safe access
std::vector<int> fwd_keys;
fwd_keys.reserve(fwd.size());
for (const auto& kv : fwd) {
    fwd_keys.push_back(kv.first);
}

#pragma omp parallel
{
    #ifdef _OPENMP
        int tid = omp_get_thread_num();
    #else
        int tid = 0;
    #endif
    auto& local_set = thread_triangles[tid];

    #pragma omp for schedule(static)
    for (size_t idx = 0; idx < fwd_keys.size(); ++idx) {
        int u = fwd_keys[idx];
        const auto& Nu = fwd.at(u);
        for (int v : Nu) {
            const auto it = fwd.find(v);
            if (it == fwd.end()) continue;
            const auto& Nv = it->second;

            std::size_t i = 0, j = 0;
            while (i < Nu.size() && j < Nv.size()) {
                int a = Nu[i];
                int b = Nv[j];
                if (a == v) { ++i; continue; }
                if (b == u) { ++j; continue; }
                if (a == b) {
                    int w = a;
                    local_set.insert(u);
                    local_set.insert(v);
                    local_set.insert(w);
                    ++i; ++j;
                }
                else if (rank_less(a, b)) {++i;}
                else {++j;}
            }
        }
    }
}
  // Merge thread-local sets
  std::unordered_set<int> in_triangle;
  for (const auto& s : thread_triangles) {
    in_triangle.insert(s.begin(), s.end());
  }
  std::vector<int> triangle_nodes(in_triangle.begin(), in_triangle.end());
  return triangle_nodes;
}

class core {
  private:
    int number_of_nodes;
    int number_of_edges;

    std::vector<int> source_nodes;
    std::vector<int> target_nodes;

    int linkage; // Currently only single linkage is implemented

    // Graph type: 0 = directed (default/tested), 1 = undirected (experimental/slower)
    int undirected;

    // Results
    std::vector<std::vector<double> > node_hierarchy; // Stores node merge events (dendrogram)
    std::vector<std::vector<int> > linknode_equivalence;       // Tracks equivalence classes at each step

  public:
    // Constructor: initializes the core object with node/edge counts, edge lists, linkage type, and graph type
    core(
      const int N,
      const int M,
      std::vector<int> source_nodes,
      std::vector<int> target_nodes,
      const int linkage,
      const int undirected
    ) {
      number_of_nodes = N;
      number_of_edges = M;
      this->source_nodes = source_nodes;
      this->target_nodes = target_nodes;
      this->linkage = linkage;
      this->undirected = undirected;
      this->node_hierarchy = std::vector<std::vector<double>>(number_of_nodes - 1, std::vector<double>(4, 0.));
    }
    ~core(){};

    // 1. Fit hierarchical clustering using a condensed distance matrix (upper triangle)
    void fit_matrix_directed(std::vector<double> &condensed_distance_matrix);
    void fit_matrix_undirected(std::vector<double> &condensed_distance_matrix);
    void fit_matrix(std::vector<double> &condensed_distance_matrix);

    // 2. Fit hierarchical clustering using an edgelist (list of [src, tgt, dist])
    void fit_edgelist_directed(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist);
    void fit_edgelist_undirected(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist);
    void fit_edgelist(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist);

    // 3. Retrieve the node hierarchy (dendrogram: merge events)
    std::vector<std::vector<double> > get_node_hierarchy();

    // 4. Retrieve linknode_equivalence classes (community tracking per step)
    std::vector<std::vector<int> > get_linknode_equivalence();
};

std::vector<std::vector<double> > core::get_node_hierarchy() {
  return node_hierarchy;
}

std::vector<std::vector<int> > core::get_linknode_equivalence() {
  return linknode_equivalence;
}

void core::fit_edgelist(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist) {
    if (undirected == 0) {
        fit_edgelist_directed(distance_edgelist, max_dist);
    } else {
        fit_edgelist_undirected(distance_edgelist, max_dist);
    }
}

void core::fit_matrix(std::vector<double> &condensed_distance_matrix) {
    if (undirected == 0) {
        fit_matrix_directed(condensed_distance_matrix);
    } else {
        fit_matrix_undirected(condensed_distance_matrix);
    }
}

void core::fit_edgelist_undirected(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist) {
    int merge_step_counter = 0;        // Number of merges performed
    int progress_carrier = 0;          // Progress reporting
    int new_cluster_idx;               // Index for new cluster formed in this step
    int idx1, idx2;                    // Indices of clusters being merged
    int k;                             // Number of link communities at this step

    std::vector<int> src1, tgt1, src2, tgt2, new_members; // Buffers for node lists
    bool eq_true = false;              // Track equivalence event
    std::vector<int> intersection_nodes;
    std::vector<int> eq_pair(2, 0);

    // Linkage matrix and cluster membership map
    std::vector<std::vector<double>> linkage_matrix;
    std::unordered_map<int, NeighborNodes> cluster_nodes_map;

    node new_node; // For node community merges

    if (source_nodes.size() != target_nodes.size()) {
        throw std::invalid_argument("Source and target vectors must have the same size.");
    }
    for (size_t i = 0; i < source_nodes.size(); ++i) {
        if (source_nodes[i] < 0 || source_nodes[i] >= number_of_nodes ||
            target_nodes[i] < 0 || target_nodes[i] >= number_of_nodes) {
            throw std::out_of_range("Source/target indices out of bounds.");
        }
    }

    // --- Prepare Distance Matrix ---
    std::vector<edge_struct> edgelist = vecvec_to_edgelist(distance_edgelist);

    // --- Hierarchical Clustering ---
    int* merge = new int[2 * (number_of_edges - 1)];
    double* height = new double[number_of_edges - 1];
    hclust_fast_edgelist(number_of_edges, edgelist, max_dist, merge, height);
    linkage_matrix = make_linkage_matrix(merge, height, number_of_edges);

    // --- Initialize Node Community List ---
    std::vector<node> merge_list(number_of_nodes);
    for (int i = 0; i < number_of_nodes; i++) {
        merge_list[i].merge = i;
        merge_list[i].members.push_back(i);
    }

    // --- Map initial link communities to their neighbors ---
    for (int i = 0; i < number_of_edges; ++i) {
        NeighborNodes lcn(source_nodes[i], target_nodes[i]);
        cluster_nodes_map[-i - 1] = lcn;
    }

    // --- Traverse Linkage Matrix to Build Hierarchy ---
    for (size_t step = 0; step < linkage_matrix.size(); ++step) {

        // Number of link communities at this step
        k = number_of_edges - static_cast<int>(step) - 1;

        new_cluster_idx = number_of_edges + step;
        idx1 = static_cast<int>(linkage_matrix[step][0]);
        idx2 = static_cast<int>(linkage_matrix[step][1]);
        double merge_height = linkage_matrix[step][2]; // Height at which clusters are merged

        // --- Merge source/target node lists for the new link community ---
        NeighborNodes merged_lcn(
          cluster_nodes_map[idx1].adjacency_list,
          cluster_nodes_map[idx2].adjacency_list
        );

        cluster_nodes_map[step+1] = merged_lcn;

        // --- Prepare node lists in triads ---
        intersection_nodes = nodes_in_triads(merged_lcn);

        // --- Merge node communities if intersection is non-trivial ---
        merge_node_communities(
            merge_list,
            intersection_nodes,
            merge_step_counter,
            k,
            merge_height,
            number_of_nodes,
            eq_pair,
            eq_true,
            node_hierarchy,
            linknode_equivalence
        );

        intersection_nodes.clear();

        // --- Clean up cluster_nodes_map to free memory ---
        cluster_nodes_map.erase(idx1);
        cluster_nodes_map.erase(idx2);

        // --- Track equivalence events ---
        if (!eq_true) {
            eq_pair[0] = k; 
            eq_pair[1] = linknode_equivalence.empty() ? number_of_nodes : linknode_equivalence.back()[1];
            linknode_equivalence.push_back(eq_pair);
        } else {
            eq_true = false;
        }
    }

    // --- Final pass: merge any remaining node communities ---
    if (merge_list.size() > 1) {
        py::print("Adding remaining node communities due to topological constraints.");
        for (size_t i = 0; i < merge_list.size() - 1; ++i) {
            node_hierarchy[merge_step_counter][0] = merge_list.back().merge;
            node_hierarchy[merge_step_counter][1] = merge_list[i].merge;
            node_hierarchy[merge_step_counter][2] = node_hierarchy[merge_step_counter - 1][2];
            node_hierarchy[merge_step_counter][3] = merge_list.back().members.size() + merge_list[i].members.size();
            merge_list.back().merge = merge_step_counter + number_of_nodes;
            merge_list.back().members.push_back(merge_list[i].members[0]);
            eq_pair[0] = k; 
            eq_pair[1] = number_of_nodes - merge_step_counter - 1;
            linknode_equivalence.push_back(eq_pair);
            merge_step_counter++;
        }
    }
    linknode_equivalence.back()[1] = 1; // Set last equivalence to 1 (root?)
    delete[] merge;
    delete[] height;
}

void core::fit_matrix_undirected(std::vector<double> &condensed_distance_matrix) {
    int merge_step_counter = 0;        // Number of merges performed
    int progress_carrier = 0;          // Progress reporting
    int new_cluster_idx;               // Index for new cluster formed in this step
    int idx1, idx2;                    // Indices of clusters being merged
    int k;                             // Number of link communities at this step

    std::vector<int> src1, tgt1, src2, tgt2, new_members; // Buffers for node lists
    bool eq_true = false;              // Track equivalence event
    std::vector<int> intersection_nodes;
    std::vector<int> eq_pair(2, 0);

    // Linkage matrix and cluster membership map
    std::vector<std::vector<double>> linkage_matrix;
    std::unordered_map<int, NeighborNodes> cluster_nodes_map;

    node new_node; // For node community merges

    if (source_nodes.size() != target_nodes.size()) {
        throw std::invalid_argument("Source and target vectors must have the same size.");
    }
    for (size_t i = 0; i < source_nodes.size(); ++i) {
        if (source_nodes[i] < 0 || source_nodes[i] >= number_of_nodes ||
            target_nodes[i] < 0 || target_nodes[i] >= number_of_nodes) {
            throw std::out_of_range("Source/target indices out of bounds.");
        }
    }

    if (condensed_distance_matrix.size() != (number_of_edges * (number_of_edges - 1)) / 2) {
        throw std::invalid_argument("distance vector must be a condensed upper triangle matrix");
    }

    // --- Prepare Distance Matrix ---
    double* condensed_distance_array = new double[(number_of_edges * (number_of_edges - 1)) / 2];
    for (size_t i = 0; i < condensed_distance_matrix.size(); i++) {
        condensed_distance_array[i] = condensed_distance_matrix[i];
    }

    // --- Hierarchical Clustering ---
    int* merge = new int[2 * (number_of_edges - 1)];
    double* height = new double[number_of_edges - 1];
    hclust_fast(number_of_edges, condensed_distance_array, linkage, merge, height);
    linkage_matrix = make_linkage_matrix(merge, height, number_of_edges);

    // --- Initialize Node Community List ---
    std::vector<node> merge_list(number_of_nodes);
    for (int i = 0; i < number_of_nodes; i++) {
        merge_list[i].merge = i;
        merge_list[i].members.push_back(i);
    }

    // --- Map initial link communities to their neighbors ---
    for (int i = 0; i < number_of_edges; ++i) {
        NeighborNodes lcn(source_nodes[i], target_nodes[i]);
        cluster_nodes_map[-i - 1] = lcn;
    }

    // --- Traverse Linkage Matrix to Build Hierarchy ---
    for (size_t step = 0; step < linkage_matrix.size(); ++step) {

        // Number of link communities at this step
        k = number_of_edges - static_cast<int>(step) - 1;

        new_cluster_idx = number_of_edges + step;
        idx1 = static_cast<int>(linkage_matrix[step][0]);
        idx2 = static_cast<int>(linkage_matrix[step][1]);
        double merge_height = linkage_matrix[step][2]; // Height at which clusters are merged

        // --- Merge source/target node lists for the new link community ---
        NeighborNodes merged_lcn(
          cluster_nodes_map[idx1].adjacency_list,
          cluster_nodes_map[idx2].adjacency_list
        );

        cluster_nodes_map[step+1] = merged_lcn;

        // --- Prepare node lists in triads ---
        intersection_nodes = nodes_in_triads(merged_lcn);

        // --- Merge node communities if intersection is non-trivial ---
        merge_node_communities(
            merge_list,
            intersection_nodes,
            merge_step_counter,
            k,
            merge_height,
            number_of_nodes,
            eq_pair,
            eq_true,
            node_hierarchy,
            linknode_equivalence
        );

        intersection_nodes.clear();

        // --- Clean up cluster_nodes_map to free memory ---
        cluster_nodes_map.erase(idx1);
        cluster_nodes_map.erase(idx2);

        // --- Track equivalence events ---
        if (!eq_true) {
            eq_pair[0] = k; 
            eq_pair[1] = linknode_equivalence.empty() ? number_of_nodes : linknode_equivalence.back()[1];
            linknode_equivalence.push_back(eq_pair);
        } else {
            eq_true = false;
        }
    }

    // --- Final pass: merge any remaining node communities ---
    if (merge_list.size() > 1) {
        py::print("Adding remaining node communities due to topological constraints.");
        for (size_t i = 0; i < merge_list.size() - 1; ++i) {
            node_hierarchy[merge_step_counter][0] = merge_list.back().merge;
            node_hierarchy[merge_step_counter][1] = merge_list[i].merge;
            node_hierarchy[merge_step_counter][2] = node_hierarchy[merge_step_counter - 1][2];
            node_hierarchy[merge_step_counter][3] = merge_list.back().members.size() + merge_list[i].members.size();
            merge_list.back().merge = merge_step_counter + number_of_nodes;
            merge_list.back().members.push_back(merge_list[i].members[0]);
            eq_pair[0] = k; 
            eq_pair[1] = number_of_nodes - merge_step_counter - 1;
            linknode_equivalence.push_back(eq_pair);
            merge_step_counter++;
        }
    }
    linknode_equivalence.back()[1] = 1; // Set last equivalence to 1 (root?)
    delete[] merge;
    delete[] height;
    delete[] condensed_distance_array;
}


void core::fit_edgelist_directed(std::vector<std::vector<double>> &distance_edgelist, const double &max_dist) {
    int merge_step_counter = 0;        // Number of merges performed
    int progress_carrier = 0;          // Progress reporting
    int new_cluster_idx;               // Index for new cluster formed in this step
    int idx1, idx2;                    // Indices of clusters being merged
    int k;                             // Number of link communities at this step

    std::vector<int> src1, tgt1, src2, tgt2, new_members; // Buffers for node lists
    bool eq_true = false;              // Track equivalence event
    std::vector<int> intersection_nodes;
    std::vector<int> eq_pair(2, 0);

    // Linkage matrix and cluster membership map
    std::vector<std::vector<double>> linkage_matrix;
    std::unordered_map<int, LinkCommunityNodes> cluster_nodes_map;

    node new_node; // For node community merges

    if (source_nodes.size() != target_nodes.size()) {
        throw std::invalid_argument("Source and target vectors must have the same size.");
    }
    for (size_t i = 0; i < source_nodes.size(); ++i) {
        if (source_nodes[i] < 0 || source_nodes[i] >= number_of_nodes ||
            target_nodes[i] < 0 || target_nodes[i] >= number_of_nodes) {
            throw std::out_of_range("Source/target indices out of bounds.");
        }
    }

    // --- Prepare Distance Matrix ---
    std::vector<edge_struct> edgelist = vecvec_to_edgelist(distance_edgelist);

    // --- Hierarchical Clustering ---
    int* merge = new int[2 * (number_of_edges - 1)];
    double* height = new double[number_of_edges - 1];
    hclust_fast_edgelist(number_of_edges, edgelist, max_dist, merge, height);
    linkage_matrix = make_linkage_matrix(merge, height, number_of_edges);

    // --- Initialize Node Community List ---
    std::vector<node> merge_list(number_of_nodes);
    for (int i = 0; i < number_of_nodes; i++) {
        merge_list[i].merge = i;
        merge_list[i].members.push_back(i);
    }

    // --- Map initial link communities to their source/target nodes ---
    for (int i = 0; i < number_of_edges; ++i) {
        LinkCommunityNodes lcn;
        lcn.source.push_back(source_nodes[i]);
        lcn.target.push_back(target_nodes[i]);
        cluster_nodes_map[-i - 1] = lcn;
    }

    // --- Traverse Linkage Matrix to Build Hierarchy ---
    for (size_t step = 0; step < linkage_matrix.size(); ++step) {

        // Number of link communities at this step
        k = number_of_edges - static_cast<int>(step) - 1;

        new_cluster_idx = number_of_edges + step;
        idx1 = static_cast<int>(linkage_matrix[step][0]);
        idx2 = static_cast<int>(linkage_matrix[step][1]);
        double merge_height = linkage_matrix[step][2]; // Height at which clusters are merged

        // --- Merge source/target node lists for the new link community ---
        LinkCommunityNodes merged_lcn;
        merged_lcn.source = cluster_nodes_map[idx1].source;
        merged_lcn.source.insert(
            merged_lcn.source.end(),
            cluster_nodes_map[idx2].source.begin(),
            cluster_nodes_map[idx2].source.end()
        );
        
        unique(merged_lcn.source);

        merged_lcn.target = cluster_nodes_map[idx1].target;
        merged_lcn.target.insert(
            merged_lcn.target.end(),
            cluster_nodes_map[idx2].target.begin(),
            cluster_nodes_map[idx2].target.end()
        );
        
        unique(merged_lcn.target);

        cluster_nodes_map[step+1] = merged_lcn;

        // --- Prepare node lists for intersection ---
        src1 = merged_lcn.source;
        tgt1 = merged_lcn.target;
        intersection_nodes = intersect(src1, tgt1);

        // --- Merge node communities if intersection is non-trivial ---
        merge_node_communities(
            merge_list,
            intersection_nodes,
            merge_step_counter,
            k,
            merge_height,
            number_of_nodes,
            eq_pair,
            eq_true,
            node_hierarchy,
            linknode_equivalence
        );

        intersection_nodes.clear();

        // --- Clean up cluster_nodes_map to free memory ---
        cluster_nodes_map.erase(idx1);
        cluster_nodes_map.erase(idx2);

        // --- Track equivalence events ---
        if (!eq_true) {
            eq_pair[0] = k; 
            eq_pair[1] = linknode_equivalence.empty() ? number_of_nodes : linknode_equivalence.back()[1];
            linknode_equivalence.push_back(eq_pair);
        } else {
            eq_true = false;
        }
    }

    // --- Final pass: merge any remaining node communities ---
    if (merge_list.size() > 1) {
        py::print("Adding remaining node communities due to topological constraints.");
        for (size_t i = 0; i < merge_list.size() - 1; ++i) {
            node_hierarchy[merge_step_counter][0] = merge_list.back().merge;
            node_hierarchy[merge_step_counter][1] = merge_list[i].merge;
            node_hierarchy[merge_step_counter][2] = node_hierarchy[merge_step_counter - 1][2];
            node_hierarchy[merge_step_counter][3] = merge_list.back().members.size() + merge_list[i].members.size();
            merge_list.back().merge = merge_step_counter + number_of_nodes;
            merge_list.back().members.push_back(merge_list[i].members[0]);
            eq_pair[0] = k; 
            eq_pair[1] = number_of_nodes - merge_step_counter - 1;
            linknode_equivalence.push_back(eq_pair);
            merge_step_counter++;
        }
    }
    linknode_equivalence.back()[1] = 1; // Set last equivalence to 1 (root?)
    delete[] merge;
    delete[] height;
}

void core::fit_matrix_directed(std::vector<double> &condensed_distance_matrix) {
    int merge_step_counter = 0;        // Number of merges performed
    int progress_carrier = 0;          // Progress reporting
    int new_cluster_idx;               // Index for new cluster formed in this step
    int idx1, idx2;                    // Indices of clusters being merged
    int k;                             // Number of link communities at this step

    std::vector<int> src1, tgt1, src2, tgt2, new_members; // Buffers for node lists
    bool eq_true = false;              // Track equivalence event
    std::vector<int> intersection_nodes;
    std::vector<int> eq_pair(2, 0);

    // Linkage matrix and cluster membership map
    std::vector<std::vector<double>> linkage_matrix;
    std::unordered_map<int, LinkCommunityNodes> cluster_nodes_map;

    node new_node; // For node community merges

    if (source_nodes.size() != target_nodes.size()) {
        throw std::invalid_argument("Source and target vectors must have the same size.");
    }
    for (size_t i = 0; i < source_nodes.size(); ++i) {
        if (source_nodes[i] < 0 || source_nodes[i] >= number_of_nodes ||
            target_nodes[i] < 0 || target_nodes[i] >= number_of_nodes) {
            throw std::out_of_range("Source/target indices out of bounds.");
        }
    }

    if (condensed_distance_matrix.size() != (number_of_edges * (number_of_edges - 1)) / 2) {
        throw std::invalid_argument("distance vector must be a condensed upper triangle matrix");
    }

    // --- Prepare Distance Matrix ---
    double* condensed_distance_array = new double[(number_of_edges * (number_of_edges - 1)) / 2];
    for (size_t i = 0; i < condensed_distance_matrix.size(); i++) {
        condensed_distance_array[i] = condensed_distance_matrix[i];
    }

    // --- Hierarchical Clustering ---
    int* merge = new int[2 * (number_of_edges - 1)];
    double* height = new double[number_of_edges - 1];
    hclust_fast(number_of_edges, condensed_distance_array, linkage, merge, height);
    linkage_matrix = make_linkage_matrix(merge, height, number_of_edges);

    // --- Initialize Node Community List ---
    std::vector<node> merge_list(number_of_nodes);
    for (int i = 0; i < number_of_nodes; i++) {
        merge_list[i].merge = i;
        merge_list[i].members.push_back(i);
    }

    // --- Map initial link communities to their source/target nodes ---
    for (int i = 0; i < number_of_edges; ++i) {
        LinkCommunityNodes lcn;
        lcn.source.push_back(source_nodes[i]);
        lcn.target.push_back(target_nodes[i]);
        cluster_nodes_map[-i - 1] = lcn;
    }

    // --- Traverse Linkage Matrix to Build Hierarchy ---
    for (size_t step = 0; step < linkage_matrix.size(); ++step) {

        // Number of link communities at this step
        k = number_of_edges - static_cast<int>(step) - 1;

        new_cluster_idx = number_of_edges + step;
        idx1 = static_cast<int>(linkage_matrix[step][0]);
        idx2 = static_cast<int>(linkage_matrix[step][1]);
        double merge_height = linkage_matrix[step][2]; // Height at which clusters are merged

        // --- Merge source/target node lists for the new link community ---
        LinkCommunityNodes merged_lcn;
        merged_lcn.source = cluster_nodes_map[idx1].source;
        merged_lcn.source.insert(
            merged_lcn.source.end(),
            cluster_nodes_map[idx2].source.begin(),
            cluster_nodes_map[idx2].source.end()
        );
        unique(merged_lcn.source);

        merged_lcn.target = cluster_nodes_map[idx1].target;
        merged_lcn.target.insert(
            merged_lcn.target.end(),
            cluster_nodes_map[idx2].target.begin(),
            cluster_nodes_map[idx2].target.end()
        );
        unique(merged_lcn.target);

        cluster_nodes_map[step+1] = merged_lcn;

        // --- Prepare node lists for intersection ---
        src1 = merged_lcn.source;
        tgt1 = merged_lcn.target;
        intersection_nodes = intersect(src1, tgt1);

        // --- Merge node communities if intersection is non-trivial ---
        merge_node_communities(
            merge_list,
            intersection_nodes,
            merge_step_counter,
            k,
            merge_height,
            number_of_nodes,
            eq_pair,
            eq_true,
            node_hierarchy,
            linknode_equivalence
        );

        intersection_nodes.clear();

        // --- Clean up cluster_nodes_map to free memory ---
        cluster_nodes_map.erase(idx1);
        cluster_nodes_map.erase(idx2);

        // --- Track equivalence events ---
        if (!eq_true) {
            eq_pair[0] = k; 
            eq_pair[1] = linknode_equivalence.empty() ? number_of_nodes : linknode_equivalence.back()[1];
            linknode_equivalence.push_back(eq_pair);
        } else {
            eq_true = false;
        }
    }

    // --- Final pass: merge any remaining node communities ---
    if (merge_list.size() > 1) {
        py::print("Adding remaining node communities due to topological constraints.");
        for (size_t i = 0; i < merge_list.size() - 1; ++i) {
            node_hierarchy[merge_step_counter][0] = merge_list.back().merge;
            node_hierarchy[merge_step_counter][1] = merge_list[i].merge;
            node_hierarchy[merge_step_counter][2] = node_hierarchy[merge_step_counter - 1][2];
            node_hierarchy[merge_step_counter][3] = merge_list.back().members.size() + merge_list[i].members.size();
            merge_list.back().merge = merge_step_counter + number_of_nodes;
            merge_list.back().members.push_back(merge_list[i].members[0]);
            eq_pair[0] = k; 
            eq_pair[1] = number_of_nodes - merge_step_counter - 1;
            linknode_equivalence.push_back(eq_pair);
            merge_step_counter++;
        }
    }
    linknode_equivalence.back()[1] = 1; // Set last equivalence to 1 (root?)
    delete[] merge;
    delete[] height;
    delete[] condensed_distance_array;
}

PYBIND11_MODULE(node_community_hierarchy_cpp, m) {
    py::class_<core>(m, "core", py::module_local())
        .def(
          py::init<
            const int,
            const int,
            std::vector<int>,
            std::vector<int>,
            const int,
            const int
          >()
        )
        .def("fit_matrix", &core::fit_matrix)
        .def("fit_edgelist", &core::fit_edgelist)
        .def("get_node_hierarchy", &core::get_node_hierarchy)
        .def("get_linknode_equivalence", &core::get_linknode_equivalence);
}
