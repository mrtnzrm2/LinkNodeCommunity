//
// C++ standalone verion of fastcluster by Daniel Muellner
//
// Copyright: Daniel Muellner, 2011
//            Christoph Dalitz, 2020
// License:   BSD style license
//            (see the file LICENSE for details)
//

#ifndef fastclustercpp_H
#define fastclustercpp_H

#include <cstdint>

typedef int_fast32_t t_index;
typedef double t_float;

// Now you can use t_index and t_float everywhere in this header
// ...

struct edge_struct {
    t_index link1;
    t_index link2;
    t_float dist;
    edge_struct(t_index l1, t_index l2, t_float d) : link1(l1), link2(l2), dist(d) {}
};


bool fc_isnan(double x);

//
// Assigns cluster labels (0, ..., nclust-1) to the n points such
// that the cluster result is split into nclust clusters.
//
// Input arguments:
//   n      = number of observables
//   merge  = clustering result in R format
//   nclust = number of clusters
// Output arguments:
//   labels = allocated integer array of size n for result
//
void cutree_k(int n, const int* merge, int nclust, int* labels);

//
// Assigns cluster labels (0, ..., nclust-1) to the n points such
// that the hierarchical clsutering is stopped at cluster distance cdist
//
// Input arguments:
//   n      = number of observables
//   merge  = clustering result in R format
//   height = cluster distance at each merge step
//   cdist  = cutoff cluster distance
// Output arguments:
//   labels = allocated integer array of size n for result
//
void cutree_cdist(int n, const int* merge, double* height, double cdist, int* labels);

//
// Hierarchical clustering with one of Daniel Muellner's fast algorithms
//
// Input arguments:
//   n       = number of observables
//   distmat = condensed distance matrix, i.e. an n*(n-1)/2 array representing
//             the upper triangle (without diagonal elements) of the distance
//             matrix, e.g. for n=4:
//               d00 d01 d02 d03
//               d10 d11 d12 d13   ->  d01 d02 d03 d12 d13 d23
//               d20 d21 d22 d23
//               d30 d31 d32 d33
//   method  = cluster metric (see enum hclust_fast_methods)
// Output arguments:
//   merge   = allocated (n-1)x2 matrix (2*(n-1) array) for storing result.
//             Result follows R hclust convention:
//              - observabe indices start with one
//              - merge[i][] contains the merged nodes in step i
//              - merge[i][j] is negative when the node is an atom
//   height  = allocated (n-1) array with distances at each merge step
// Return code:
//   0 = ok
//   1 = invalid method
//
int hclust_fast(int n, double* distmat, int method, int* merge, double* height);


//
// Hierarchical clustering using a memory-efficient single-linkage algorithm.
//
// This method is analogous to hclust_fast, but uses the MeMST_linkage_core
// algorithm, which operates on a sparse edge list instead of a condensed
// distance matrix. This allows efficient clustering of large, sparse networks
// where only neighboring links (edges) and their dissimilarities are stored.
//
// Input arguments:
//   n       = number of observables (nodes/links to be clustered)
//   edges   = vector of edge_struct, each encoding a pair of neighboring links
//             and their dissimilarity (distance)
//   max_dist= maximum allowed distance for merging disconnected components;
//             merges at this value ensure a rooted dendrogram
//   merge   = allocated (n-1)x2 matrix (2*(n-1) array) for storing result.
//             Result follows R hclust convention:
//              - observable indices start with one
//              - merge[i][] contains the merged nodes in step i
//              - merge[i][j] is negative when the node is an atom
//   height  = allocated (n-1) array with distances at each merge step
//
// Return code:
//   0 = ok
//   1 = error (e.g., invalid input)
//
// The output is compatible with cutree_k, cutree_cdist, and other dendrogram
// utilities. This method is recommended for sparse graphs where the majority
// of link pairs are not neighbors.
//
int hclust_fast_MeMSINGLE(int n, const std::vector<edge_struct>& edges, t_float max_dist, int* merge, double* height);

//
// Constructs the (n-1)x4 linkage matrix in R format from merge/height arrays
//
// Input arguments:
//   merge  = clustering result in R format (2*(n-1) array)
//   height = cluster distance at each merge step (n-1 array)
//   n      = number of observables
// Output:
//   returns (n-1)x4 linkage matrix (vector of vector<double>)
//   Each row: [merge1, merge2, height, cluster_size]
//
std::vector<std::vector<double>> make_linkage_matrix(const int* merge, const double* height, int n);

enum hclust_fast_methods {
  // single link with the minimum spanning tree algorithm (Rohlf, 1973)
  HCLUST_METHOD_SINGLE = 0,
  // complete link with the nearest-neighbor-chain algorithm (Murtagh, 1984)
  HCLUST_METHOD_COMPLETE = 1,
  // unweighted average link with the nearest-neighbor-chain algorithm (Murtagh, 1984)
  HCLUST_METHOD_AVERAGE = 2,
  // median link with the generic algorithm (Müllner, 2011)
  // requires euclidean distances as distance data
  HCLUST_METHOD_MEDIAN = 3,
  HCLUST_METHOD_WARD = 4,
  HCLUST_METHOD_WEIGHTED = 5,
  HCLUST_METHOD_CENTROID = 6,
};
  

#endif
