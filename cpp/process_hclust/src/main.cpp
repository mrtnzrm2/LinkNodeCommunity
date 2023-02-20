#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include "hclust-cpp/fastcluster.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double mean(std::vector<double> &v) {
  if (v.size() == 0) {
    printf("Warning: mean of vector with zero size\n");
    return 0;
  }
  double mv = 0;
  for (int i = 0; i < v.size(); i++)
    mv += v[i];
  return mv / v.size();
}

double sum(std::vector<double> &v) {
  if (v.size() == 0) {
    printf("Warning: mean of vector with zero size\n");
    return 0;
  }
  double mv = 0;
  for (int i = 0; i < v.size(); i++)
    mv += v[i];
  return mv;
}

template<typename T>
void unique(std::vector<T> &v) {
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

template<typename T>
void print_v(std::vector<T> &v) {
  for (int i = 0; i < v.size(); i++)
    printf("%i ", (int) v[i]);
  printf("\n");
}

template<typename T>
void quotient_set(
  std::vector<T> &a,std::vector<T> &b, std::vector<T> &result
) {
  // result = a / b
  bool exist;
  for (int i = 0; i < a.size(); i++) {
    exist = false;
    for (int j = 0; j <= b.size() / 2; j++) {
      if (a[i] == b[j] || a[i] == b[b.size() - j - 1])
        exist = true;
    }
    if (!exist)
      result.push_back(a[i]);
  }
}

template<typename T>
void intersection(
  std::vector<T> &a, std::vector<T> &b, std::vector<T> &result
) {
  std::set_intersection(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
  );
}

void vec_times_k(std::vector<int>& v, int k) {
    std::transform(v.begin(), v.end(), v.begin(), [k](int &c){ return c*k; });
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
    [&v](size_t i1, size_t i2) {return v[i1] < v[i2];}
  );

  return idx;
}

double get_muscore(
  std::vector<int> lc, double &ntrees, double &k, int &linkage,
  int& size, int& M, int& n, double& beta
) {
  // Order lc ----
  vec_times_k(lc, -1);
  std::sort(lc.begin(), lc.end());
  vec_times_k(lc, -1);
  double mu = 0;
  double D;
  if (size >= n) {
    for (int i=0; i < n - 1; i++) {
      for (int j=(i + 1); j < n; j++) {
        D = static_cast<double>(lc[j]) / static_cast<double>(lc[i]) ;
        if (D > beta)
          mu += D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
        else
          mu -= D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
      }
    }
    mu /= 0.5 * n * (n - 1);
  } else {
    for (int i=0; i < size - 1; i++) {
      for (int j=(i + 1); j < size; j++) {
         D = static_cast<double>(lc[j]) / static_cast<double>(lc[i]) ;
        if (D > beta)
          mu += D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
        else
          mu -= D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
      }
    }
    mu /= 0.5 * size * (size - 1);
  }
  if (ntrees == 0 && linkage == 2) mu *= k * k;
  return mu;
}

double approx_nodes_by_edges(int& M) {
  double x;
  double m = static_cast<double>(M);
  double x1 = (1 + sqrt(1 + 4 * m)) / 2;
  x1 = floor(x1);
  double x2 = (1 - sqrt(1 + 4 * m)) / 2;
  x2 = floor(x2);
  if (x2 <= 0)
    x = x1;
  else
    x = x2;
  return x;
}

void get_sizes(
  int* labels,
  std::vector<int>& lcs,
  std::vector<int>& nds,
  std::vector<int>& unique_labels,
  std::vector<int>& source,
  std::vector<int>& target,
  int& n
) {

  std::vector<std::vector <int> > node_buffer(unique_labels.size());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < unique_labels.size(); j++) {
      if (labels[i] == unique_labels[j]) {
        lcs[j]++;
        node_buffer[j].push_back(source[i]);
        node_buffer[j].push_back(target[i]);
      }
    }
  }
  for (int j = 0; j < unique_labels.size(); j++) {
    unique(node_buffer[j]);
    nds[j] = node_buffer[j].size();
  }
}

std::vector<double> simplify_height_to_k_end(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int* size
) {
  double h = height[0];
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    if (i < n - 2) {
      if (height[i + 1] != h) {
        sim_k.push_back(n - i);
        sim_height.push_back(h);
        h = height[i + 1];
        ++(*size);
      }
    } else {
      if (height[i] != height[i - 1]) {
        sim_k.push_back(n - i);
        h = height[i];
        sim_height.push_back(h);
        ++(*size);
      }
    }
    
  }
  return sim_k;
}

std::vector<double> simplify_height_to_k_start(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int* size
) {
  double h = height[0];
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    if (i == 0) {
      sim_k.push_back(n - 1);
      sim_height.push_back(h);
      ++(*size);
    }
    if (height[i] != h && i != 0) {
      h = height[i];
      sim_k.push_back(n - i - 1);
      sim_height.push_back(h);
      ++(*size);
    }
  }
  return sim_k;
}

std::vector<double> complete_height_to_k(
  int &n,
  double* height,
  std::vector<double>& sim_height,
  int* size
) {
  std::vector<double> sim_k;
  for (int i = 0; i < n - 1; i++) {
    sim_k.push_back(n - i - 1);
    sim_height.push_back(height[i]);
    ++(*size);
}
  return sim_k;
}

void get_dc(
  std::vector<int> &source,
  std::vector<int> &target,
  int* labels, int& lc_id, int& n,
  double &dc
) {
  double m = 0, N;
  std::vector<int> nodes;
  for (int i = 0; i < n; i++) {
    if (labels[i] == lc_id) {
      m++;
      nodes.push_back(source[i]);
      nodes.push_back(target[i]);
    }
  }
  // uniques ----
  unique(nodes);
  N = nodes.size();
  dc = static_cast<double>(m - N + 1) /
    static_cast<double>(pow(N - 1, 2));
}

int get_nac(
  std::vector<int> &source,
  std::vector<int> &target,
  int* labels, int& lc_id, int& n
) {
  int N;
  std::vector<int> nodes_src;
  std::vector<int> nodes_tgt;
  for (int i = 0; i < n; i++) {
    if (labels[i] == lc_id) {
      if (source[i] > target[i]) nodes_src.push_back(source[i]);
      if (source[i] < target[i]) nodes_tgt.push_back(target[i]);
    }
  }
  // uniques ----
  unique(nodes_src);
  unique(nodes_tgt);
  // intersect
  std::vector<int> inter;
  intersection(nodes_src, nodes_tgt, inter);
  N = inter.size();
  return N;
}

double get_percolation_susceptability(
  std::vector<int> lc_sizes,
  std::vector<int> lc_number_of_nodes,
  int& number_of_lcs
) {
  double N = 0;
  double  x = 0; // percolation suceptability
  std::vector<int> new_lc_sizes(lc_sizes);
  sort(
    new_lc_sizes.begin(), new_lc_sizes.end(), std::greater<int>()
  );
  for (int i = 0; i < number_of_lcs; i++) {
    N += lc_sizes[i];
  }
  N = pow(N, 2);
  for (int i = 0; i < number_of_lcs; i++) {
    if (lc_sizes[i] > 1 && lc_number_of_nodes[i] > 2 && lc_sizes[i] != new_lc_sizes[0])
      x += pow(lc_sizes[i], 2) / N;
  }
  return x;
}

template<typename T>
double order_parameter(std::vector<T> & v, int &M) {
  double n = v.size();
  double m = 0;
  for (int i=0; i < n; i++) {
    if (m < v[i]) m = v[i];
  }
  m /= M;
  return m;
}

struct tree {
  int size;
  int count;
};

template<typename T>
double Xm(std::vector<T> & v) {
  double n = v.size();
  double xm2 = 0, xm = 0;
  // Number of clusters of size s
  tree new_tree;
  std::vector<tree> ns;
  bool check_size_exist;
  int size_index;
  for (int i=0; i < n; i++) {
    check_size_exist = false;
    for (int j=0; j < ns.size(); j++) {
      if (ns[j].size == v[i]) {
        check_size_exist = true;
        size_index = j;
        break;
      }
    }
    if (check_size_exist) {
      ns[size_index].count++;
    }
    else {
      new_tree.size = v[i];
      new_tree.count = 1;
      ns.push_back(new_tree);
    }
  }
  // Sum over the number of different link community sizes
  for (int i=0; i < ns.size(); i++) {
    xm2 += pow(ns[i].size, 2.0) * ns[i].count;
    xm += ns[i].size * ns[i].count;
  }
  return xm2 / xm;
}

std::vector<std::vector<double> > process_hclust_fast(
  int& n,
  std::vector<std::vector<double> >& distmat,
  std::vector<int>& source,
  std::vector<int>& target,
  int& nodes,
  int& linkage,
  bool& cut,
  int& nss,
  double& beta
)
{
  // Various variables ----
  int nac, number_lcs, nec;
  double dc;
  // Condense distance matrix ----
  double* tri_distmat = new double[(n*(n-1))/2];
  // hclust arrays ----
  int* merge = new int[2*(n-1)];
  double* height = new double[n-1];
  // Reduced K and heigths ----
  std::vector<double> sim_k, sim_height;
  // Dc vector ----
  std::vector<double> dcv;
  // Effective number of merging steps ----
  int* K = new int;
  *K = 0;
  // labels pointer ----
  int* labels = new int[n];
  /////////////////////
  // Get condense matrix ----
  for (int i = 0, k = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      tri_distmat[k] = distmat[i][j];
      k++;
    }
  }
  // Get hierarchy!! ----
  hclust_fast(
    n,
    tri_distmat,
    linkage, // linkage method
    merge,
    height
  );
  if (cut) {
    // Delete duplicated heights preserving the
    // first K and height ----
    sim_k = simplify_height_to_k_start(n, height, sim_height, K);
  } else {
    // Keep complete the all the steps ----
    sim_k = complete_height_to_k(n, height, sim_height, K);
  }
  // Matrix for hierarchical features ----
  // 0: K
  // 1: Height
  // 2: NAC
  // 3: NEC
  // 4: mu
  // 5: D
  // 6: ntrees
  // 7: Percolation susceptability (X)
  // 8: Order parameters m(t)
  // 9: Susceptibility xm(t)
  std::vector<std::vector<double> > features_main(
    *K, std::vector<double>(10, 0)
  );
  // THE GAME STARTS
  for (int i=0; i < *K; i++) {
    // Assign height ----
    // K
    features_main[i][0] = sim_k[i];
    // Height
    features_main[i][1] = sim_height[i];
    // NAC
    features_main[i][2] = nodes;
    // Cut tree at given sim_k and get
    // memberships ----
    cutree_k(
      n,
      merge,
      sim_k[i],
      labels
    );
    std::vector<int> unique_labels(labels, labels + n);
    unique(unique_labels);
    number_lcs = unique_labels.size();
    // Get LCs sizes in order
    std::vector<int> lc_size(number_lcs, 0);
    std::vector<int> node_size(number_lcs, 0);
    get_sizes(
      labels,
      lc_size, node_size,
      unique_labels, source, target, n
    );
    features_main[i][8] = order_parameter(lc_size, n);
    features_main[i][9] = Xm(lc_size);
    nec = 0;
    // Loop over link communities ----
    for (int j=0; j < number_lcs; j++) {
      // Check if the community is a tree ----
      if (lc_size[j] > 1 && node_size[j] > 2) {
        nac = get_nac(
          source, target,
          labels, unique_labels[j],
          n
        );
        get_dc(
          source, target, labels,
          unique_labels[j], n, dc
        );

        dcv.push_back(dc * lc_size[j] / n);
        // NAC
        if (nac >= 1) features_main[i][2] -= (nac - 1);
        // ntrees
        if (dc <= 0) features_main[i][6]++;
        nec++;
      } else {
        dcv.push_back(0);
      }
    }
    // Mu-score
    features_main[i][4] = get_muscore(
      lc_size, features_main[i][6], sim_k[i], linkage,
      number_lcs, n, nss, // ss average range
      beta
    );
    // NEC
    features_main[i][3] = nec;
    // D
    features_main[i][5] = sum(dcv);
    // X
    features_main[i][7] = get_percolation_susceptability(
      lc_size, node_size, number_lcs
    );
    dcv.clear();
  }
  // Delete phase
  delete[] labels;
  delete K;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
  // Return
  return features_main;
}

int get_k_bad(
  int& n,
  std::vector<std::vector<double> >& distmat,
  int& linkage
) {
   // From distance matrix to upper triangular array
  double* tri_distmat = new double[(n*(n-1))/2];
  for (int i = 0, k = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      tri_distmat[k] = distmat[i][j];
      k++;
    }
  }
  // hclust
  int* merge = new int[2*(n-1)];
  double* height = new double[n-1];
  hclust_fast(
    n,
    tri_distmat,
    linkage, // linkage method
    merge,
    height
  );
  // simplify height to k
  std::vector<double> k;
  std::vector<double> sim_height;
  int* K = new int;
  *K = 0;
  k = simplify_height_to_k_start(n, height, sim_height, K);
  return *K;
}

PYBIND11_MODULE(process_hclust, m) {

  m.doc() = "pybind11 process hclust";

  m.def(
    "process_hclust_fast",
    &process_hclust_fast,
    py::return_value_policy::reference_internal
  );

  m.def(
    "get_k_bad",
    &get_k_bad,
    py::return_value_policy::reference_internal
  );
}