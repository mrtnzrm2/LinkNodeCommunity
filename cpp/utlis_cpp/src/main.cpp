/*
 * cpp/utlis_cpp/src/main.cpp
 *
 * Library: utils_cpp
 * Author: Jorge S. Martinez Armas
 *
 * Overview:
 * ----------
 * Lightweight utilities shared across the link-node package C++ extensions.
 * - Convert an MST, provided as sparse link–link distances, into a SciPy‑compatible linkage.
 * - Provide fast similarity kernels used by Python bindings.
 *
 * Exposed Functions (pybind11):
 * -----------------------------
 * - mst_edges_to_linkage(int N,
 *     const std::vector<std::vector<double>>& edges,
 *     double max_dist) -> std::vector<std::vector<double>>
 *   Inputs:
 *     N: number of links (M in other modules).
 *     edges: list of [link1, link2, dist] with 1‑based link indices; symmetric distances.
 *     max_dist: distance used when components are disconnected.
 *   Output:
 *     (N-1) x 4 linkage matrix [i, j, height, size] in SciPy format.
 *
 * - hellinger_similarity_graph(std::vector<double>& fu,
 *     std::vector<double>& fv, int& iu, int& iv) -> double
 *   Computes an approximation to the Bhattacharyya coefficient between
 *   two non‑negative feature vectors, treating entries at positions iu and iv
 *   as cross‑paired.
 *   Requirements: fu.size() == fv.size(), sums > 0, 0 <= iu,iv < N.
 *
 * - cosine_similarity_graph(std::vector<double>& u,
 *     std::vector<double>& v, int& ii, int& jj) -> double
 *   Cosine similarity of u and v, excluding positions ii and jj from the
 *   dot product and then cross‑pairing those two entries (u[ii] with v[jj]
 *   and u[jj] with v[ii]). Requirements: u.size() == v.size(), non‑zero norms,
 *   0 <= ii,jj < N.
 *
 * Implementation Notes:
 * ---------------------
 * - mst_edges_to_linkage uses a Prim‑style pass on the sparse edge map and
 *   then converts to R‑style merge arrays before emitting the SciPy linkage.
 * - Indices in 'edges' are 1‑based to match the rest of the project; internal
 *   computations convert to 0‑based.
 * - Exceptions are thrown as std::invalid_argument for invalid inputs.
 *
 * Usage:
 * ------
 * Import from Python as 'utils_cpp' and call the functions above. See README for examples.
 */

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include<ctime> // time

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <tuple>
#include <unordered_map>
#include <set>
#include <limits>
#include <cassert>
#include <stdexcept>

// ClusterResult definition
class ClusterResult {
public:
    ClusterResult(int N) : N_(N) {}
    void append(int node1, int node2, double dist) {
        Z_.emplace_back(node1, node2, dist);
    }
    std::tuple<int, int, double> operator[](size_t idx) const {
        return Z_[idx];
    }
    size_t size() const {
        return Z_.size();
    }
    std::vector<std::tuple<int, int, double>>& data() { return Z_; }
    const std::vector<std::tuple<int, int, double>>& data() const { return Z_; }
private:
    std::vector<std::tuple<int, int, double>> Z_;
    int N_;
};

// Print contents of Z2.Z for debugging
void print_Z2Z(const ClusterResult& Z2) {
    const auto& Z = Z2.data();
    std::cout << "Z2.Z (debug print):" << std::endl;
    for (size_t i = 0; i < Z.size(); ++i) {
        int node1 = std::get<0>(Z[i]);
        int node2 = std::get<1>(Z[i]);
        double dist = std::get<2>(Z[i]);
        std::cout << "step " << i << ": (" << node1 << ", " << node2 << ", " << dist << ")" << std::endl;
    }
}

// C++ equivalent of Python's UnionFind
class UnionFind {
public:
    UnionFind(int size)
        : parent_(2 * size - 1, 0), nextparent_(size) {}

    int find(int idx) {
        if (parent_[idx] != 0) {
            int p = idx;
            idx = parent_[idx];
            while (parent_[idx] != 0) {
                idx = parent_[idx];
            }
            while (parent_[p] != idx) {
                int tmp = parent_[p];
                parent_[p] = idx;
                p = tmp;
            }
        }
        return idx;
    }

    void unite(int node1, int node2) {
        parent_[node1] = parent_[node2] = nextparent_;
        ++nextparent_;
    }

    int nextparent_idx() const { return nextparent_; }

private:
    std::vector<int> parent_;
    int nextparent_;
};

// C++ equivalent to generate_R_dendrogram
void generate_R_dendrogram(const ClusterResult& Z2, int N, std::vector<int>& merge, std::vector<double>& height, bool sorted = false) {
    // Z2: ClusterResult containing (node1, node2, dist)
    // N: number of nodes
    // merge: output, size 2*(N-1)
    // height: output, size N-1
    // sorted: whether Z2 is already sorted by dist

    std::vector<std::tuple<int, int, double>> Z2_sorted = Z2.data();
    if (!sorted) {
        std::stable_sort(Z2_sorted.begin(), Z2_sorted.end(),
                 [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
    }

    merge.resize(2 * (N - 1), 0);
    height.resize(N - 1, 0.0);
    std::vector<int> node_size(N - 1, 0);

    UnionFind nodes(N);

    for (int i = 0; i < N - 1; ++i) {
        int node1, node2;
        double dist;
        if (sorted) {
            std::tie(node1, node2, dist) = Z2_sorted[i];
        } else {
            node1 = nodes.find(std::get<0>(Z2_sorted[i]));
            node2 = nodes.find(std::get<1>(Z2_sorted[i]));
            nodes.unite(node1, node2);
            dist = std::get<2>(Z2_sorted[i]);
        }
        if (node1 > node2) std::swap(node1, node2);

        int size1, size2;
        // For node1
        if (node1 < N) {
            merge[i] = -node1 - 1;
            size1 = 1;
        } else {
            merge[i] = node1 - N + 1;
            size1 = node_size[node1 - N];
        }
        // For node2
        if (node2 < N) {
            merge[i + N - 1] = -node2 - 1;
            size2 = 1;
        } else {
            merge[i + N - 1] = node2 - N + 1;
            size2 = node_size[node2 - N];
        }
        height[i] = dist;
        node_size[i] = size1 + size2;
    }
}

// Helper for memst_linkage_core: edge_map typedef
using EdgeMap = std::unordered_map<int, std::unordered_map<int, double>>;

// Accepts edges as std::vector<std::vector<double>> where each edge is {link1, link2, dist}
ClusterResult memst_linkage_core(int N, const std::vector<std::vector<double>>& edges, double max_dist) {
    // Build edge lookup map for O(1) access
    std::unordered_map<int, std::unordered_map<int, double>> edge_map;
    for (const auto& edge : edges) {
        int link1 = static_cast<int>(edge[0]) - 1;
        int link2 = static_cast<int>(edge[1]) - 1;
        double dist = edge[2];
        edge_map[link1][link2] = dist;
        edge_map[link2][link1] = dist;
    }

    // Active nodes are those not yet in the tree
    std::set<int> active_nodes;
    for (int i = 1; i < N; ++i) active_nodes.insert(i);

    // Best known distance to the current tree and the corresponding parent
    std::vector<double> d(N, max_dist);
    std::vector<int> best_src(N, -1);

    // Initialize distances and parents from start node 0
    auto it0 = edge_map.find(0);
    for (int v = 1; v < N; ++v) {
        best_src[v] = 0;  // default to 0; updated when a better edge is found
        if (it0 != edge_map.end()) {
            auto jt = it0->second.find(v);
            if (jt != it0->second.end()) {
                d[v] = jt->second;
            }
        }
    }

    // Find first node to attach to node 0
    int idx2 = -1;
    double min_dist = max_dist;
    for (int v : active_nodes) {
        if (d[v] < min_dist) {
            min_dist = d[v];
            idx2 = v;
        }
    }
    if (idx2 == -1 && !active_nodes.empty()) {
        idx2 = *active_nodes.begin();
        min_dist = d[idx2];
    }

    ClusterResult Z2(N - 1);
    if (idx2 != -1) {
        Z2.append(best_src[idx2], idx2, min_dist);
        active_nodes.erase(idx2);
    }

    // Main loop: Prim's algorithm
    for (int step = 1; step < N - 1; ++step) {
        int prev_node = idx2;

        // Relax edges from the most recently added node
        auto edge_it = edge_map.find(prev_node);
        for (int node : active_nodes) {
            if (edge_it != edge_map.end()) {
                auto dist_it = edge_it->second.find(node);
                if (dist_it != edge_it->second.end() && dist_it->second < d[node]) {
                    d[node] = dist_it->second;
                    best_src[node] = prev_node;
                }
            }
        }

        // Pick the next node with minimal connection distance
        idx2 = -1;
        min_dist = max_dist;
        for (int node : active_nodes) {
            if (d[node] < min_dist) {
                min_dist = d[node];
                idx2 = node;
            }
        }
        if (idx2 == -1 && !active_nodes.empty()) {
            idx2 = *active_nodes.begin();
            min_dist = d[idx2];
        }

        Z2.append(best_src[idx2], idx2, min_dist);
        active_nodes.erase(idx2);
    }
    return Z2;
}


// r_to_idx helper
inline int r_to_idx(int r, int n) {
    return r < 0 ? -r - 1 : r + n - 1;
}

/**
 * Construct the linkage matrix in SciPy format from R-style merge and height arrays,
 * using union-find to track ancestry mapping.
 *
 * @param merge R-style merge array, negative for leaves, positive for merged clusters. Size: 2*(n-1)
 * @param height Heights at each merge step. Size: n-1
 * @param n Number of initial leaves.
 * @return Linkage matrix in SciPy format: (n-1) x 4, [idx1, idx2, height, sample_count]
 */
std::vector<std::vector<double>> make_linkage_matrix_scipy(const std::vector<int>& merge,
                                                           const std::vector<double>& height,
                                                           int n) {
    std::vector<std::vector<double>> linkage(n-1, std::vector<double>(4, 0.0));
    // Size bookkeeping for clusters: initially all leaves are size 1
    std::vector<int> cluster_sizes(2 * n - 1, 1);
    UnionFind uf(n);

    for (int step = 0; step < n-1; ++step) {
        int m1 = merge[step];
        int m2 = merge[step + n - 1];
        int idx1 = r_to_idx(m1, n);
        int idx2 = r_to_idx(m2, n);

        // Find current representatives for each cluster using union-find
        int root1 = uf.find(idx1);
        int root2 = uf.find(idx2);

        // Compute new cluster index
        int new_cluster_idx = uf.nextparent_idx();
        // Cluster sizes
        int size1 = cluster_sizes[root1];
        int size2 = cluster_sizes[root2];
        int merged_size = size1 + size2;
        cluster_sizes[new_cluster_idx] = merged_size;

        // Output linkage row, smaller index first
        int i1 = std::min(root1, root2);
        int i2 = std::max(root1, root2);
        linkage[step][0] = i1;
        linkage[step][1] = i2;
        linkage[step][2] = height[step];
        linkage[step][3] = merged_size;

        // Union the clusters in the union-find structure
        uf.unite(root1, root2);
    }

    return linkage;
}


// -- The requested function --
std::vector<std::vector<double>> mst_edges_to_linkage(int N, const std::vector<std::vector<double>>& edges, double max_dist) {
    ClusterResult Z2 = memst_linkage_core(N, edges, max_dist);

    std::vector<int> merge;
    std::vector<double> height;
    generate_R_dendrogram(Z2, N, merge, height, false);

    std::vector<std::vector<double>> linkage = make_linkage_matrix_scipy(merge, height, N);

    return linkage;
}


double hellinger_similarity_graph(
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

double cosine_similarity_graph(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
    if (ii < 0 || jj < 0 || ii >= N || jj >= N) {
        throw std::invalid_argument("Indices ii and jj must be within vector bounds.");
    }

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

double tanimoto_coefficient_graph(
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

double pearson_correlation_graph(
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

double weighted_jaccard_graph(
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

double jaccard_probability_graph(
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

PYBIND11_MODULE(utils_cpp, m) {

    m.doc() = "Assisting C++ tools";
    m.def(
        "mst_edges_to_linkage",
        &mst_edges_to_linkage,
        py::return_value_policy::reference_internal
    );
    m.def(
        "hellinger_similarity_graph",
        &hellinger_similarity_graph,
        py::return_value_policy::reference_internal
    );
    m.def(
        "cosine_similarity_graph",
        &cosine_similarity_graph,
        py::return_value_policy::reference_internal
    );
    m.def(
        "pearson_correlation_graph",
        &pearson_correlation_graph,
        py::return_value_policy::reference_internal
    );
    m.def(
        "tanimoto_coefficient_graph",
        &tanimoto_coefficient_graph,
        py::return_value_policy::reference_internal
    );
    m.def(
        "weighted_jaccard_graph",
        &weighted_jaccard_graph,
        py::return_value_policy::reference_internal
}
