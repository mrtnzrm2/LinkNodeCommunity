#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include "hclust-cpp/fastcluster.h"
#include "entropy_tools.cpp"
#include "bene_tools.cpp"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct lcprops {
  int m = 0;
  int n = 0;
};

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
void intersection(
  std::vector<T> &a, std::vector<T> &b, std::vector<T> &result
) {
  std::set_intersection(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
  );
}

double get_muscore(
  std::vector<int> &sizes, double &K, int& M, int& alpha, double& beta
) {
  double mu = 0;
  double D;
  if (K >= alpha) {
    for (int i=0; i < alpha - 1; i++) {
      for (int j=i + 1; j <alpha; j++) {
        D = sizes[j]/ (sizes[i] * 1.) ;
        if (D > beta)
          mu += D * (sizes[j] + sizes[i]) / (2. * M);
        else
          mu -= D * (sizes[j] + sizes[i]) / (2. * M);
      }
    }
    mu /= 0.5 * alpha * (alpha - 1);
  } else if (K >= 2) {
    for (int i=0; i < K - 1; i++) {
      for (int j=i + 1; j < K; j++) {
         D = sizes[j] /(sizes[i] * 1.)  ;
        if (D > beta)
          mu += D * (sizes[i] + sizes[j]) / (2. * M);
        else
          mu -= D * (sizes[i] + sizes[j]) / (2. * M);
      }
    }
    mu /= 0.5 * K * (K - 1);
  } else {
    mu = 0.;
  }
  return mu;
}

bool search_key(std::map<int, std::vector<int> > &a, const int &key) {
  for (std::map<int, std::vector<int> >::iterator f=a.begin(); f != a.end(); ++f) {
    if (f->first == key) return true;
  }
  return false;
}

bool search_key(std::map<int, lcprops > &a, const int &key) {
  for (std::map<int, lcprops>::iterator f=a.begin(); f != a.end(); ++f) {
    if (f->first == key) return true;
  }
  return false;
}

bool cmp( std::pair<int, std::vector<int> > &a, std::pair <int, std::vector<int> > &b ){
   return a.second[0] > b.second[0];
}

std::map<int, std::vector<int>> sort_map(std::map<int, std::vector<int>> givenMap){
   std::vector<std::pair<int, std::vector<int> > > pairVec;
   std::map<int, std::vector<int>> newMap;
   for ( auto& it : givenMap ) {
      pairVec.push_back( it );
   }
   sort( pairVec.begin(), pairVec.end(), cmp); 
   for ( auto& it : pairVec ) {
      newMap.insert( { it.first, it.second } );
   }
   return newMap;
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

double Dc(
  int&m, int& n
) {
  double dc;
  dc = static_cast<double>(m - n + 1) /
    static_cast<double>(pow(n- 1, 2));
  return dc;
}

double get_percolation_susceptability(std::map<int, lcprops> &v, int &N, int &order) {
  double  x = 0; // percolation suceptability
  for (std::map<int, lcprops >::iterator it = v.begin(); it != v.end(); ++it) {
    if (it->second.m <= 1 || it->second.n <= 2) continue;
    if (it->second.m != order)
      x += pow(it->second.m, 2.);
  }
  return x / N;
}

double order_parameter(std::vector<int> &v, int &M) {
  return v[0] / (M * 1.);
}

double Xm(std::map<int, lcprops> &v) {
  double n = v.size();
  double xm2 = 0, xm = 0;
  std::map<int, int> v_count;
  for (std::map<int, lcprops>::iterator it = v.begin(); it != v.end(); ++it) {
    if (!search_key(v, it->second.m)) v_count[it->second.m] = 1;
    else v_count[it->second.m]++;
  }
  for (std::map<int, int >::iterator it = v_count.begin(); it != v_count.end(); ++it) {
    xm2 += pow(it->first, 2.0) * it->second;
    xm += it->first * it->second * 1.;
  }
  if (xm > 0)
    return xm2 / xm;
  else return 0;
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
    int max_level=0;

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
    // std::vector<std::vector<int> > &link_communities, std::vector<double> & H, 
    void arbre(std::string &t_size);
    void bene(std::string &t_size);

    template <typename T>
    void expand_vector(std::vector<T>& v, const int& N);

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
    int get_max_level();
    void get_sizes(
      std::map<int, lcprops> &info_sizes,
      int* labels, std::vector<int> &lcsize,
      std::vector<int>& unique_labels,
      std::vector<int>& source,
      std::vector<int>& target,
      int& n
    );
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
void ph::expand_vector(std::vector<T>& v, const int& N) {
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

int ph::get_max_level(){
  return max_level;
}

void ph::bene(std::string &t_size) {
  std::vector<double> H(number_of_elements, 0);
  std::vector<std::vector<double> >  bene(6, std::vector<double>(number_of_elements - 1, 0.));
  std::vector<std::vector<int> > link_communities(number_of_elements, std::vector<int>(number_of_elements, 0));
  std::map<std::string, vertex_properties_bene> tree;
  std::map<int, tracer_properties > tracer;
  //
  expand_vector(K, number_of_elements - 1);
  expand_vector(Height, number_of_elements - 1);
  expand_vector(D, number_of_elements - 1);
  expand_vector(MU, number_of_elements - 1);
  expand_vector(NEC, number_of_elements - 1);
  expand_vector(ntrees, number_of_elements - 1);
  expand_vector(X, number_of_elements - 1);
  expand_vector(XM, number_of_elements - 1);
  expand_vector(OrP, number_of_elements - 1);
  //
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  int* labels = new int[number_of_elements];
  // Get condense matrix ----
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
  // Get heights ----
  for (int i=0; i < number_of_elements - 1; i++)
    H[i+1] = height[i];
  Z2dict_bene(link_communities, tree, tracer, H, source_vertices, target_vertices, t_size);
  BUONO(tracer, tree, bene, ALPHA, BETA, number_of_elements);

  for (int i = 0; i < number_of_elements - 1; i++) {
    NEC[i] = bene[0][i];
    ntrees[i] = bene[3][i];
    Height[i] = height[i];
    K[i] = number_of_elements - i - 1;
    MU[i] = bene[1][i];
    D[i] = bene[2][i];
    X[i] = bene[4][i];
    XM[i] = bene[4][i];
    OrP[i] = bene[5][i];
  }
  // Delete pointers
  delete[] labels;
  delete[] merge;
  delete[] height;
  delete[] tri_distmat;
}

// std::vector<std::vector<int> > &link_communities, std::vector<double> & H,
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
  // Get heights ----
  for (int i=0; i < number_of_elements - 1; i++)
    H[i+1] = height[i];

  // std::cout << H[number_of_elements - 2] << " " << H[number_of_elements - 1] << "\n";

  // for (auto j : H) {
  //   std::cout << j << " ";
  // }
  // std::cout << "\n";
  
  std::map<int, level_properties> chain;
  std::cout << "Starting Z2dict\n";
  std::map<std::string, vertex_properties> tree;
  Z2dict(link_communities, tree, H, t_size);

  // for (int i = 0; i < link_communities.size(); i++) {
  //   for (int j = 0; j < link_communities[i].size(); j++) {
  //     std::cout << link_communities[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // for (std::map<std::string, vertex_properties>::iterator it = tree.begin(); it != tree.end(); ++it) {
  //   std::cout << it->first << "\t\t\t" << it->second.level << "\t" << it->second.height << std::endl;
  //   // for (std::set<std::string>::iterator ii = it->second.post_key.begin(); ii != it->second.post_key.end(); ii++) {
  //   //   std::cout << *ii << "\t\t";
  //   // }
  //   // std::cout << "\n\n";
  // }

  std::cout << "Level information\n";
  level_information(tree, root, chain);

  // for (std::map<int, level_properties>::iterator v = chain.begin(); v != chain.end(); ++v){
  //   std::cout << v->first << "\t" << v->second.size << "\t" << v->second.height << "\n";
  // }

  // Get max level ----
  for (std::map<int, level_properties>::iterator it = chain.begin(); it != chain.end(); ++it) {
    if (it->first > max_level)
     max_level = it->first;
  }
  // level_information_H(tree, root, chain_h, max_level);

  // for (std::map<int, level_properties>::iterator v = chain_h.begin(); v != chain_h.end(); ++v){
  //   std::cout << v->first << "\t" << v->second.size << "\t" << v->second.height << "\n";
  // }
  // std::cout << number_of_elements << "\n";

  std::cout << "Vertex entropy\n";
  vertex_entropy(tree, chain, root, number_of_elements, Sh);
  std::cout << "Vertex entropy H\n";
  vertex_entropy_H(tree, chain, root, number_of_elements, max_level, ShH);
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

void ph::get_sizes(
  std::map<int, lcprops> &info_sizes,
  int* labels, std::vector<int> &lcsize,
  std::vector<int>& unique_labels,
  std::vector<int>& source,
  std::vector<int>& target,
  int& n
) {
  std::vector<std::set<int> > node_buffer(unique_labels.size());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < unique_labels.size(); j++) {
      if (labels[i] == unique_labels[j]) {
        info_sizes[unique_labels[j]].m++;
        lcsize[j]++;
        node_buffer[j].insert(source[i]);
        node_buffer[j].insert(target[i]);
      }
    }
  }
  for (int j = 0; j < unique_labels.size(); j++)
    info_sizes[unique_labels[j]].n = node_buffer[j].size();
  sort(lcsize.begin(), lcsize.end(), std::greater<int>());
}

void ph::vite() {
  // Various variables ----
  int nt, nec;
  double dc;
  std::vector<double> sim_k, sim_height, dcv;
  std::vector<int> lcsizes;
  std::map<int, lcprops > sizes;
  // Condense distance matrix ----
  double* tri_distmat = new double[(number_of_elements * (number_of_elements - 1)) / 2];
  // hclust arrays ----
  int* merge = new int[2 * (number_of_elements - 1)];
  double* height = new double[number_of_elements-1];
  // Effective number of merging steps ----
  int* kk = new int;
  *kk = 0;
  int* labels = new int[number_of_elements];
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
    lcsizes = std::vector<int>(unique_labels.size(), 0);
    // Get LCs sizes in order
    get_sizes(sizes, labels, lcsizes, unique_labels, source_vertices, target_vertices, number_of_elements);
    nec = 0;
    nt = 0;
    // Loop over link communities ----
    for (std::map<int, lcprops >::iterator it=sizes.begin(); it != sizes.end(); ++it) {
      if (it->second.m > 1 && it->second.n > 2) {
        dc = Dc(it->second.m, it->second.n);
        dcv.push_back(dc * it->second.m / number_of_elements);
        if (dc <= 0) nt++;
        nec++;
      } else {
        dcv.push_back(0);
      }
    }
    ntrees[i] = nt;
    // Mu-score
    MU[i] = get_muscore(lcsizes, sim_k[i], number_of_elements, ALPHA, BETA);
    // NEC
    NEC[i] = nec;
    // D
    D[i] = sum(dcv);
    // Order parameter
    OrP[i] = order_parameter(lcsizes, number_of_elements);
    // XM
    XM[i] = Xm(sizes);
    // X
    X[i] = get_percolation_susceptability(sizes, number_of_elements, lcsizes[0]);
    dcv.clear();
    sizes.clear();
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
        .def("bene", &ph::bene)
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
        .def("get_entropy_v_H", &ph::get_entropy_v_H)
        .def("get_max_level", &ph::get_max_level);
}