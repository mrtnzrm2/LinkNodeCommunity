#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>

struct vertex_properties {
  int level;
  double height;
  std::set<std::string> post_key;
};

struct level_properties {
  int size;
  double height;
};

template<typename T, typename U>
double stirling_3(T Nl, U N) {
  return  log((static_cast<double>(Nl) / static_cast<double>(N)) * (pow(2. * M_PI * static_cast<double>(Nl), 1. / (2. * static_cast<double>(Nl))) / pow(2. * M_PI * static_cast<double>(N), 1. / (2. * static_cast<double>(N))))) + ((1./12.) * (pow(static_cast<double>(N), -2.)- pow(static_cast<double>(Nl), -2.)));
}

template<typename T, typename U>
double stirling_1(T Nl, U N) {
  return  log((static_cast<double>(Nl) / static_cast<double>(N)) );
}

template <typename T>
void unique_2(std::vector<T> &v) {
  std::sort(v.begin(), v.end());
  std::vector<int>::iterator it;
  it = std::unique(v.begin(), v.end());  
  v.resize(std::distance(v.begin(), it));  
}

std::set<int> intersection(std::set<int> &a, std::set<int> &b) {
  std::set<int> intersect;
  std::set_intersection(
    a.begin(), a.end(), b.begin(), b.end(),
    std::inserter(intersect, intersect.begin())
  );
  return intersect;
}

std::set<int> where(std::vector<int> &v, int &com) {
  std::set<int> where_com;
  for (int i=0; i < v.size(); i++) {
    if (v[i] == com) {
      where_com.insert(i);
    }
  }
  return where_com;
}

std::set<int> where(std::vector<int> &v, std::set<int> &pred,const int &com) {
  std::set<int> where_com;
  for (std::set<int>::iterator it = pred.begin(); it != pred.end(); ++it) {
    if (v[*it] == com) {
      where_com.insert(*it);
    }
  }
  return where_com;
}

bool search_key(std::map<std::string, vertex_properties> &a, std::string &key) {
  for (std::map<std::string, vertex_properties>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first.compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::map<std::string, vertex_properties> &a, const std::string &key) {
  for (std::map<std::string, vertex_properties>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first.compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::map<int, int> &a, int &key) {
  for (std::map<int, int>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first == key) return true;
  }
  return false;
}

bool search_key(std::map<int, level_properties> &a, int &key) {
  for (std::map<int, level_properties>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first == key) return true;
  }
  return false;
}

bool search_key(std::vector<std::string> &a, std::string &key) {
  for (std::vector<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::vector<std::string> &a, const std::string &key) {
  for (std::vector<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::vector<std::string> &a, const char * key) {
  for (std::vector<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::set<std::string> &a, std::string &key) {
  for (std::set<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::set<std::string> &a, const std::string &key) {
  for (std::set<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

bool search_key(std::set<std::string> &a, const char * key) {
  for (std::set<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
    if ((*it).compare(key) == 0) return true;
  }
  return false;
}

int number_nodes(std::set<int> &v, std::vector<int> &src, std::vector<int> &tar) {
  std::set<int> L;
  for (std::set<int>::iterator it = v.begin(); it != v.end(); ++it) {
    L.insert(src[*it]);
    L.insert(tar[*it]);
  }
  return L.size();
}

std::set<int> unique_with_pred(std::vector<int> &a, std::set<int> & pred) {
  std::set<int> s;
  for (std::set<int>::iterator v = pred.begin(); v != pred.end(); ++v) {
    s.insert(a[*v]);
  }
  return s;
}

void Z2dict_short(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  const std::string key_pred, std::set<int> nodes_pred, std::vector<double> &H, const int L, const int tL, const int &nodes
) {
  if (L < A.size() && nodes_pred.size() > 1) {
    const int next_L = L + 1, next_tL = tL + 1;
    std::set<int> coms = unique_with_pred(A[L], nodes_pred);
    std::set<int> nodes_com;
    std::set<int> compare;
    for (std::set<int>::iterator com = coms.begin(); com != coms.end(); ++com) {
      const std::string key = "L" + std::to_string(next_tL) + std::to_string(*com);
      nodes_com = where(A[L], nodes_pred, *com);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() == 0) continue;
      if (nodes_com.size() == nodes_pred.size()) {
        Z2dict_short(A, tree, key_pred, nodes_com, H, next_L, tL, nodes);
      }
      else if (nodes_com.size() < nodes_pred.size()) {
        if (!search_key(tree, key_pred)) {
          // std::cout << key_pred << " " << L << "\n";
          tree[key_pred].level = tL;
          tree[key_pred].height = H[nodes - L];
          tree[key_pred].post_key.insert(key_pred + key);
        }
        else {
          tree[key_pred].post_key.insert(key_pred + key);
        }
        Z2dict_short(A, tree, key_pred + key, nodes_com, H, next_L, next_tL, nodes);
      }
    }
  } else {
    if (!search_key(tree, key_pred)) {
      tree[key_pred].level = tL;
      tree[key_pred].height = H[nodes - L];
      tree[key_pred].post_key.insert("END");
    }
  }
}

void Z2dict_short_2(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  const std::string key_pred, std::set<int> nodes_pred, std::vector<double> &H, const int L, const int tL, const int &nodes
) {
  if (L < A.size() && nodes_pred.size() > 1) {
    int next_L = L + 1, next_tL = tL + 1;
    std::set<int> coms = unique_with_pred(A[L], nodes_pred);
    std::set<int> nodes_com;
    std::set<int> compare;
    for (std::set<int>::iterator com = coms.begin(); com != coms.end(); ++com) {
      const std::string key = "L" + std::to_string(L) + std::to_string(*com);
      nodes_com = where(A[L], nodes_pred, *com);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() == 0) continue;
      if (nodes_com.size() == nodes_pred.size()) {
        Z2dict_short_2(A, tree, key_pred, nodes_com, H, next_L, tL, nodes);
      }
      else if (nodes_com.size() < nodes_pred.size()) {
        if (!search_key(tree, key_pred)) {
          tree[key_pred].level = L - 1;
          tree[key_pred].height = H[nodes - L];
          tree[key_pred].post_key.insert(key_pred + key);
        }
        else {
          tree[key_pred].post_key.insert(key_pred + key);
        }
        Z2dict_short_2(A, tree, key_pred + key, nodes_com, H, next_L, next_tL, nodes);
      }
    }
  } else {
    if (!search_key(tree, key_pred)) {
      tree[key_pred].level = L - 1;
      tree[key_pred].height = H[nodes - L];
      tree[key_pred].post_key.insert("END");
    } else {
      tree[key_pred].post_key.insert("END");
    }
  }
}

void Z2dict_long(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  const std::string key_pred, std::set<int> nodes_pred, std::vector<double> &H, const int L, const int tL, const int &nodes
) {
  // Probably needs to be checked
  if (L < A.size() && nodes_pred.size() > 1) {
    int next_L = L + 1, next_tL = tL + 1;
    std::set<int> coms = unique_with_pred(A[L], nodes_pred);
    std::set<int> nodes_com;
    std::set<int> compare;
    for (std::set<int>::iterator com = coms.begin(); com != coms.end(); ++com) {
      const std::string key = "L" + std::to_string(next_tL) + std::to_string(*com);
      nodes_com = where(A[L], nodes_pred, *com);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() > 0) {
        if (!search_key(tree, key_pred)) {
          tree[key_pred].level = L - 1;
          tree[key_pred].height = H[nodes - L];
          tree[key_pred].post_key.insert(key_pred + key);
        }
        else {
          tree[key_pred].post_key.insert(key_pred + key);
        }
        Z2dict_long(A, tree, key_pred + key, nodes_com, H, next_L, next_tL, nodes);
      }
    }
  } else {
    if (!search_key(tree, key_pred)) {
      tree[key_pred].level = L - 1;
      tree[key_pred].height = H[nodes - L];
      tree[key_pred].post_key.insert("END");
    }
  }
}

void Z2dict(std::vector<std::vector<int> > &A,  std::map<std::string, vertex_properties> &tree, std::vector<double> &H, std::string &type) {
  const int N = A.size();
  std::set<int> nodes;
  for (int i=0; i < A.size(); i++) {
    nodes.insert(A[A.size() - 1][i]);
  }
  const int L = 1, tL = 0;
  const std::string root = "L00";
  if (type.compare("short") == 0) Z2dict_short(A, tree, root, nodes, H, L, tL, N);
  else if (type.compare("short_2") == 0) Z2dict_short_2(A, tree, root, nodes, H, L, tL, N);
  else if (type.compare("long") == 0) Z2dict_long(A, tree, root, nodes, H, L, tL, N);
  else {
    throw std::runtime_error("\nOnly types: short, short_2 or long\n");
  }
}

void sum_vertices(std::map<std::string, vertex_properties> &tree, const std::string &root, int &t) {
  for (std::set<std::string>::iterator s = tree[root].post_key.begin(); s != tree[root].post_key.end(); ++s) {
    if ((*s).compare("END") == 0) continue;
    t++;
    sum_vertices(tree, *s, t);
  }
}

void level_information(std::map<std::string, vertex_properties> &tree, const std::string &root, std::map<int, level_properties> &ml) {
  if (!search_key(tree[root].post_key, "END")) {
    auto it1 = next(tree[root].post_key.begin(), 0);
    auto it2 = next(tree[root].post_key.begin(), 1);
    if (!search_key(ml, tree[root].level)) {
      ml[tree[root].level].size = 1;
      ml[tree[root].level].height = (2 * tree[root].height - tree[*it1].height - tree[*it2].height) / 2.;
    } else {
      ml[tree[root].level].size++;
      ml[tree[root].level].height += (2 * tree[root].height - tree[*it1].height - tree[*it2].height) / 2.;
    }
    for (std::set<std::string>::iterator v = tree[root].post_key.begin(); v != tree[root].post_key.end(); ++v) {
      level_information(tree, *v, ml);
    }
  } 
  else {
    if (!search_key(ml, tree[root].level)) {
      ml[tree[root].level].size = 1;
      ml[tree[root].level].height = 0; //tree[root].height;
    }
    else {
      ml[tree[root].level].size++;
      ml[tree[root].level].height += 0; //tree[root].height;
    }
  }
}

void level_information_H(std::map<std::string, vertex_properties> &tree, const std::string &root, std::map<int, level_properties> &ml, int &maxlvl) {
  if (!search_key(tree[root].post_key, "END")) {
    if (!search_key(ml, tree[root].level)) {
      ml[tree[root].level].size = 1;
      ml[tree[root].level].height = 0;
     for (std::set<std::string>::iterator v = tree[root].post_key.begin(); v != tree[root].post_key.end(); ++v) {
         ml[tree[root].level].height += (tree[root].height - tree[*v].height) / 2.;
         level_information_H(tree, *v, ml, maxlvl);
      } 
    } else {
      ml[tree[root].level].size++;
      for (std::set<std::string>::iterator v = tree[root].post_key.begin(); v != tree[root].post_key.end(); ++v) {
         ml[tree[root].level].height += (tree[root].height - tree[*v].height) / 2.;
         level_information_H(tree, *v, ml, maxlvl);
      } 
    }
  } 
  else {
    if (!search_key(ml, tree[root].level)) {
      ml[tree[root].level].size = 1;
      ml[tree[root].level].height = 0; //tree[root].height;
    }
    else {
      ml[tree[root].level].size++;
      ml[tree[root].level].height += 0; //tree[root].height;
    }
  }
}

void vertex_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, const std::string &root, const int &nodes, std::vector<double> &Sh) {
  if (!search_key(tree[root].post_key, "END")) {
    int nextL = tree[root].level + 1, tL = tree[root].level;
    double Mul = tree[root].post_key.size();
    Sh[nodes - tL - 1] -= Mul * stirling_3(Mul, static_cast<double>(ml[nextL].size));
    for (std::set<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo)
      vertex_entropy(tree, ml, *roo, nodes, Sh);
  }
}

void level_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, const std::string &root, const int &nodes, std::vector<double> &sv) {
  int M = 1;
  sum_vertices(tree, root, M);
  // std::cout << "Number of vertices in the tree: " << M << "\n";
  for (std::map<int, level_properties>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    // Vertical entropy
    sv[nodes - ll->first - 1] -= ll->second.size * stirling_3(static_cast<double>(ll->second.size), M);
  }
}

void vertex_entropy_H(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> ml, const std::string &root, const int &nodes, int &max_levels, std::vector<double> &Sh) {
  double Mul = tree[root].post_key.size();
  int nextL = tree[root].level + 1, tL = tree[root].level;
  if (!search_key(tree[root].post_key, "END")) {
    for (std::set<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      Sh[nodes - tL - 1] -= (tree[root].height - tree[*roo].height) * stirling_3(Mul, static_cast<double>(ml[nextL].size)) / 2.;
      vertex_entropy_H(tree, ml, *roo, nodes, max_levels, Sh);
    }
  }
}

void level_entrop_H(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, const std::string &root, const int &nodes, std::vector<double> &sv) {
  int M = 1;
  sum_vertices(tree, root, M);
  std::cout << "Number of vertices in the tree: " << M << "\n";
  for (std::map<int, level_properties>::iterator ll = ml.begin(); ll != ml.end(); ++ll)
    sv[nodes - ll->first - 1] -= ll->second.height * stirling_3(static_cast<double>(ll->second.size), M);
}