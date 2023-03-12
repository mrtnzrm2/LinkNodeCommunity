#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include "hclust-cpp/fastcluster.h"
#include "entropy_tools.cpp"

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
  std::vector<int> lc, int &ntrees, double &k, int &linkage,
  int& size, int& M, int& alpha, double& beta
) {
  // Order lc ----
  vec_times_k(lc, -1);
  std::sort(lc.begin(), lc.end());
  vec_times_k(lc, -1);
  double mu = 0;
  double D;
  if (size >= alpha) {
    for (int i=0; i < alpha - 1; i++) {
      for (int j=(i + 1); j < alpha; j++) {
        D = static_cast<double>(lc[j]) / static_cast<double>(lc[i]) ;
        if (D > beta)
          mu += D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
        else
          mu -= D * (static_cast<double>(lc[i] + lc[j]) / static_cast<double>(2 * M));
      }
    }
    mu /= 0.5 * alpha * (alpha - 1);
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

void get_sizes(
  int* labels,
  std::vector<int>& lcs,
  std::vector<int>& nds,
  std::vector<int>& unique_labels,
  std::vector<int>& source,
  std::vector<int>& target,
  int& n
) {

  std::vector<std::set<int> > node_buffer(unique_labels.size());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < unique_labels.size(); j++) {
      if (labels[i] == unique_labels[j]) {
        lcs[j]++;
        node_buffer[j].insert(source[i]);
        node_buffer[j].insert(target[i]);
      }
    }
  }
  for (int j = 0; j < unique_labels.size(); j++) {
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
  std::set<int> nodes;
  for (int i = 0; i < n; i++) {
    if (labels[i] == lc_id) {
      m++;
      nodes.insert(source[i]);
      nodes.insert(target[i]);
    }
  }
  N = nodes.size();
  dc = static_cast<double>(m - N + 1) /
    static_cast<double>(pow(N - 1, 2));
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
  // for (int i = 0; i < number_of_lcs; i++) {
    // N += lc_sizes[i];
  // }
  // N = pow(N, 2);
  for (int i = 0; i < number_of_lcs; i++) {
    if (lc_sizes[i] <= 1 || lc_number_of_nodes[i] <= 2) continue;
    if (lc_sizes[i] != new_lc_sizes[0])
      x += pow(lc_sizes[i], 2);
    N += lc_sizes[i];
  }
  return x / N;
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

class ph {
  private:

    std::vector<int> K;
    std::vector<double> Height;
    std::vector<int> NEC;
    std::vector<double> MU;
    std::vector<double> D;
    std::vector<int> ntrees;
    std::vector<double> X;
    std::vector<double> OrP;
    std::vector<double> XM;

    std::vector<double> Sh;
    std::vector<double> Sv;
    std::vector<double> ShH;
    std::vector<double> SvH;

    int number_of_elements;
    std::vector<std::vector<double> > distane_matrix;
    std::vector<int> source_vertices;
    std::vector<int> target_vertices;
    int total_nodes;
    int Linkage;
    bool CUT;
    int ALPHA;
    double BETA;

  public:
    ph(
      const int n,
      std::vector<std::vector<double> > distmat,
      std::vector<int> source,
      std::vector<int> target,
      const int nodes,
      const int linkage,
      const bool cut,
      const int alpha,
      const double beta
    );
    ~ph(){};

    void vite();

    void arbre(std::string &t_size);

    template <typename T>
    void expand_vector(std::vector<T>& v, int& N);

    std::vector<int> get_K();
    std::vector<double> get_Height();
    std::vector<int> get_NEC();
    std::vector<double> get_MU();
    std::vector<double> get_D();
    std::vector<int> get_ntrees();
    std::vector<double> get_X();
    std::vector<double> get_OrP();
    std::vector<double> get_XM();
    std::vector<double> get_entropy_h();
    std::vector<double> get_entropy_v();
    std::vector<double> get_entropy_h_H();
    std::vector<double> get_entropy_v_H();
};

ph::ph(
  const int n,
  std::vector<std::vector<double> > distmat,
  std::vector<int> source,
  std::vector<int> target,
  const int nodes,
  const int linkage,
  const bool cut,
  const int alpha,
  const double beta
) {
  number_of_elements = n;
  distane_matrix = distmat;
  source_vertices = source;
  target_vertices = target;
  total_nodes = nodes;
  Linkage = linkage;
  CUT = cut;
  ALPHA = alpha;
  BETA = beta;
}

template <typename T>
void ph::expand_vector(std::vector<T>& v, int& N) {
  v = std::vector<T>(N, 0);
}

std::vector<int> ph::get_K() {
  return K;
}
std::vector<double> ph::get_Height() {
  return Height;
}
std::vector<int> ph::get_NEC() {
  return NEC;
}
std::vector<double> ph::get_MU() {
  return MU;
}
std::vector<double> ph::get_D() {
  return D;
}
std::vector<int> ph::get_ntrees() {
  return ntrees;
}
std::vector<double> ph::get_X() {
  return X;
}
std::vector<double> ph::get_OrP() {
  return OrP;
}
std::vector<double> ph::get_XM() {
  return XM;
}
std::vector<double> ph::get_entropy_h(){
  return Sh;
}

std::vector<double> ph::get_entropy_v(){
  return Sv;
}

std::vector<double> ph::get_entropy_h_H(){
  return ShH;
}

std::vector<double> ph::get_entropy_v_H(){
  return SvH;
}

void ph::arbre(std::string &t_size) {
  const std::string root = "L00";
  std::vector<double> H(number_of_elements, 0);
  expand_vector(Sh, number_of_elements);
  expand_vector(Sv, number_of_elements);
  expand_vector(ShH, number_of_elements);
  expand_vector(SvH, number_of_elements);
  std::vector<std::vector<int> > link_communities(number_of_elements, std::vector<int>(number_of_elements, 0));
  // Get hierarchy!! ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  int* labels = new int[number_of_elements];
  for (int i = 0, k = 0; i < number_of_elements; i++) {
    for (int j = i +1 ; j < number_of_elements; j++) {
      tri_distmat[k] = distane_matrix[i][j];
      k++;
    }
  }
  hclust_fast(
    number_of_elements,
    tri_distmat,
    Linkage,
    merge,
    height
  );
  // Get link community matrix ----
  for (int i=1; i <= number_of_elements - 1; i++) {
    cutree_k(number_of_elements, merge, i, labels);
    for (int j=0; j < number_of_elements; j++) link_communities[i-1][j] = labels[j];
  }
  for (int i=0; i < number_of_elements; i++) {
    link_communities[number_of_elements - 1][i] = i;
  }
  for (int i=0; i < number_of_elements - 1; i++)
    H[i+1] = height[i];

  std::cout << H[number_of_elements - 2] << " " << H[number_of_elements - 1] << "\n";
  // for (auto j : H) {
  //   std::cout << j << " ";
  // }
  // std::cout << "\n";
  
  std::map<int, level_properties> chain;
  std::cout << "Starting Z2dict\n";
  std::map<std::string, vertex_properties> tree;
  Z2dict(link_communities, tree, source_vertices, target_vertices, H, t_size);

  // for (int i = 0; i < link_communities.size(); i++) {
  //   for (int j = 0; j < link_communities[i].size(); j++) {
  //     std::cout << link_communities[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // for (std::map<std::string, vertex_properties>::iterator it = tree.begin(); it != tree.end(); ++it) {
  //   std::cout << it->first << "\t\t\t" << it->second.level << "\t" << it->second.height << std::endl;

  //   for (std::set<std::string>::iterator ii = it->second.post_key.begin(); ii != it->second.post_key.end(); ii++) {
  //     std::cout << *ii << "\t\t";
  //   }
  //   std::cout << "\n\n";
  // }

  std::cout << "Level information\n";
  level_information(tree, root, chain);

  // for (std::map<int, level_properties>::iterator v = chain.begin(); v != chain.end(); ++v){
  //   std::cout << v->first << "\t" << v->second.size << "\t" << v->second.height << "\n";
  // }
  // std::cout << number_of_elements << "\n";

  std::cout << "Vertex entropy\n";

  // std::cout << root << "\n";

  vertex_entropy(tree, chain, root, number_of_elements, Sh);
  std::cout << "Vertex entropy H\n";
  vertex_entropy_H(tree, chain, root, number_of_elements, ShH);
  std::cout << "Level entropy\n";
  level_entropy(tree, chain, root, number_of_elements, Sv);
  std::cout << "Level entropy H\n";
  level_entrop_H(tree, chain, root, number_of_elements, SvH);

  // Delete pointers
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

void ph::vite() {
  // Various variables ----
  int nt, number_lcs, nec;
  double dc;
  // Condense distance matrix ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  // hclust arrays ----
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  // Reduced K and heigths ----
  std::vector<double> sim_k, sim_height;
  // Dc vector ----
  std::vector<double> dcv;
  // Effective number of merging steps ----
  int* kk = new int;
  *kk = 0;
  // labels pointer ----
  int* labels = new int[number_of_elements];
  /////////////////////
  // Get condense matrix ----
  for (int i = 0, k = 0; i < number_of_elements; i++) {
    for (int j = i +1 ; j < number_of_elements; j++) {
      tri_distmat[k] = distane_matrix[i][j];
      k++;
    }
  }
  // Get hierarchy!! ----
  hclust_fast(
    number_of_elements,
    tri_distmat,
    Linkage, // linkage method
    merge,
    height
  );
  if (CUT) {
    // Delete duplicated heights preserving the
    // first K and height ----
    sim_k = simplify_height_to_k_start(number_of_elements, height, sim_height, kk);
  } else {
    // Keep complete the all the steps ----
    sim_k = complete_height_to_k(number_of_elements, height, sim_height, kk);
  }
  expand_vector(K, *kk);
  expand_vector(Height, *kk);
  expand_vector(D, *kk);
  expand_vector(MU, *kk);
  expand_vector(NEC, *kk);
  expand_vector(ntrees, *kk);
  expand_vector(X, *kk);
  expand_vector(XM, *kk);
  expand_vector(OrP, *kk);
  // THE GAME STARTS
  for (int i=0; i < *kk; i++) {
    // Assign height ----
    // K
    K[i] = sim_k[i];
    // Height
    Height[i] = sim_height[i];
    // Cut tree at given sim_k and get
    // memberships ----
    cutree_k(
      number_of_elements,
      merge,
      sim_k[i],
      labels
    );
    std::vector<int> unique_labels(labels, labels + number_of_elements);
    unique(unique_labels);
    number_lcs = unique_labels.size();
    // Get LCs sizes in order
    std::vector<int> lc_size(number_lcs, 0);
    std::vector<int> node_size(number_lcs, 0);
    get_sizes(
      labels, lc_size, node_size, unique_labels, source_vertices, target_vertices, number_of_elements
    );
    nec = 0;
    nt = 0;
    // Loop over link communities ----
    for (int j=0; j < number_lcs; j++) {
      // Check if the community is a tree ----
      if (lc_size[j] > 1 && node_size[j] > 2) {
        get_dc(
          source_vertices, target_vertices, labels,
          unique_labels[j], number_of_elements, dc
        );
        dcv.push_back(dc * lc_size[j] / number_of_elements);
        // ntrees
        if (dc <= 0) nt++;
        nec++;
      } else {
        dcv.push_back(0);
      }
    }
    // Number of trees
    ntrees[i] = nt;
    // Mu-score
    MU[i] = get_muscore(
      lc_size, nt, sim_k[i], Linkage,
      number_lcs, number_of_elements, ALPHA, BETA
    );
    // NEC
    NEC[i] = nec;
    // D
    D[i] = sum(dcv);
    // Order parameter
    OrP[i] = order_parameter(lc_size, number_of_elements);
    // XM
    XM[i] = Xm(lc_size);
    // X
    X[i] = get_percolation_susceptability(
      lc_size, node_size, number_lcs
    );
    dcv.clear();
  }
  // Delete phase
  delete[] labels;
  delete kk;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

PYBIND11_MODULE(process_hclust, m) {
    py::class_<ph>(m, "ph")
        .def(
          py::init<
            const int,
            std::vector<std::vector<double> >,
            std::vector<int>,
            std::vector<int>,
            const int,
            const int,
            const bool,
            const int,
            const double
          >()
        )
        .def("vite", &ph::vite)
        .def("arbre", &ph::arbre)
        .def("get_K", &ph::get_K)
				.def("get_Height", &ph::get_Height)
				.def("get_NEC", &ph::get_NEC)
        .def("get_MU", &ph::get_MU)
				.def("get_D", &ph::get_D)
			  .def("get_ntrees", &ph::get_ntrees)
        .def("get_X", &ph::get_X)
        .def("get_OrP", &ph::get_OrP)
			  .def("get_XM", &ph::get_XM)
        .def("get_entropy_h", &ph::get_entropy_h)
        .def("get_entropy_v", &ph::get_entropy_v)
        .def("get_entropy_h_H", &ph::get_entropy_h_H)
        .def("get_entropy_v_H", &ph::get_entropy_v_H);
}