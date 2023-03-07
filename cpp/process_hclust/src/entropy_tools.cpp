#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>

struct vertex_properties {
  int level;
  int edges;
  int nodes;
  std::vector<std::string> post_key;
};

struct level_properties {
  int total_cls;
  int max_size_cls;
  std::map<int, int> sizes_cls;
};

template <typename T>
void unique_tree(std::vector<T> &v) {
  std::sort(v.begin(), v.end());
  std::vector<int>::iterator it;
  it = std::unique(v.begin(), v.end());  
  v.resize(std::distance(v.begin(),it));  
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

bool search_key(std::map<std::string, vertex_properties> &a, std::string &key) {
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

bool search_key(std::vector<std::string> &a, const char * key) {
  for (std::vector<std::string>::iterator it = a.begin(); it != a.end(); ++it) {
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

void Z2dict_short(
  std::vector<std::vector<int> > &A, std::vector<int> &src, std::vector<int> &tar, std::map<std::string, vertex_properties> &tree,
  std::string key_pred, std::set<int> nodes_pred, int L, int tL
) {
  if (L < A.size()) {
    int next_L = L + 1, next_tL = tL + 1;
    std::vector<int> coms = A[L];
    unique_tree(coms);
    std::string key;
    std::set<int> nodes_com;
    std::set<int> compare;
    for (int i=0; i < coms.size(); i++) {
      key = "L" + std::to_string(tL + 1) + std::to_string(i);
      nodes_com = where(A[L], coms[i]);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() == 0) continue;
      if (nodes_com.size() == nodes_pred.size()) {
        Z2dict_short(A, src, tar, tree, key_pred, nodes_com, next_L, tL);
      }
      else if (nodes_com.size() < nodes_pred.size()) {
        if (!search_key(tree, key_pred)) {
          vertex_properties item;
          item.level = tL;
          item.edges = nodes_pred.size();
          item.nodes = number_nodes(nodes_pred, src, tar);
          item.post_key.push_back(key_pred + "_" + key);
          tree.insert(std::pair<std::string, vertex_properties>(key_pred, item));
        }
        else {
          tree[key_pred].post_key.push_back(key_pred + "_" + key);
        }
        Z2dict_short(A, src, tar, tree, key_pred + "_" + key, nodes_com, next_L, next_tL);
      }
    }
  } else {
    tree[key_pred].level = tL;
    tree[key_pred].edges = 1;
    tree[key_pred].nodes = 2;
    tree[key_pred].post_key.push_back("END");
  }
}

void Z2dict_long(
  std::vector<std::vector<int> > &A, std::vector<int> &src, std::vector<int> &tar, std::map<std::string, vertex_properties> &tree,
  std::string key_pred, std::set<int> nodes_pred, int L, int tL
) {
  if (L < A.size()) {
    int next_L = L + 1, next_tL = tL + 1;
    std::vector<int> coms = A[L];
    unique_tree(coms);
    std::string key;
    std::set<int> nodes_com;
    std::set<int> compare;
    for (int i=0; i < coms.size(); i++) {
      key = "L" + std::to_string(tL + 1) + std::to_string(i);
      nodes_com = where(A[L], coms[i]);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() > 0) {
        if (!search_key(tree, key_pred)) {
          vertex_properties item;
          item.level = tL;
          item.edges = nodes_pred.size();
          item.nodes = number_nodes(nodes_pred, src, tar);
          item.post_key.push_back(key_pred + "_" + key);
          tree.insert(std::pair<std::string, vertex_properties>(key_pred, item));
        }
        else {
          tree[key_pred].post_key.push_back(key_pred + "_" + key);
        }
        Z2dict_long(A, src, tar, tree, key_pred + "_" + key, nodes_com, next_L, next_tL);
      }
    }
  } else {
    tree[key_pred].level = tL;
    tree[key_pred].edges = 1;
    tree[key_pred].nodes = 2;
    tree[key_pred].post_key.push_back("END");
  }
}

std::map<std::string, vertex_properties> Z2dict(std::vector<std::vector<int> > &A, std::vector<int> &src, std::vector<int> &tar, std::string &type) {
  std::set<int> nodes;
  for (int i=0; i < A.size(); i++) {
    nodes.insert(A[A.size() - 1][i]);
  }
  std::map<std::string, vertex_properties> tree;
  int L = 1, tL = 0;
  std::string root = "L00";
  if (type.compare("short") == 0) Z2dict_short(A, src, tar, tree, root, nodes, L, tL);
  else if (type.compare("long") == 0) Z2dict_long(A, src, tar, tree, root, nodes, L, tL);
  else {
    throw std::runtime_error("\nOnly types: short or long\n");
  }
  return tree;
}

void sum_vertices(std::map<std::string, vertex_properties> &tree, std::string &root, int &t) {
  for (std::vector<std::string>::iterator s = tree[root].post_key.begin(); s != tree[root].post_key.end(); ++s) {
    if ((*s).compare("END") == 0) continue;
    t++;
    sum_vertices(tree, *s, t);
  }
}

void level_information(std::map<std::string, vertex_properties> &tree, std::string &root, std::map<int, level_properties> &ml) {
  for (std::map<std::string, vertex_properties>::iterator t = tree.begin(); t != tree.end(); ++t) {
    if (!search_key(ml, t->second.level)) {
      level_properties item;
      item.total_cls = 1;
      item.max_size_cls = t->second.edges;
      item.sizes_cls.insert(std::pair<int, int>(t->second.edges, 1));
      ml.insert(std::pair<int, level_properties>(t->second.level, item));
    } else {
      ml[t->second.level].total_cls++;
      if (t->second.edges > ml[t->second.level].max_size_cls) ml[t->second.level].max_size_cls = t->second.edges;
      if (!search_key(ml[t->second.level].sizes_cls, t->second.edges)) {
        ml[t->second.level].sizes_cls.insert(std::pair<int, int>(t->second.edges, 1));
      } else {
        ml[t->second.level].sizes_cls[t->second.edges]++;
      }
    }
  }
}

void vertex_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> ml, std::string &root, int &nodes, std::vector<double> &Sh, std::vector<double> &Dc, std::vector<double> &x, std::vector<int> &nec, std::vector<int> &nt) {
  double Mul, dc;
  int nextL = tree[root].level + 1, tL = tree[root].level;
  if (!search_key(tree[root].post_key, "END")) {
    // Horizontal Entropy
    Mul = tree[root].post_key.size();
    Sh[nodes - tL - 1] -= Mul * log(Mul / static_cast<double>(ml[nextL].total_cls)) / nodes;
    // Density
    dc = (tree[root].edges - tree[root].nodes + 1) / (pow(tree[root].nodes - 1, 2.));
    Dc[nodes - tL - 1] += dc * tree[root].edges / nodes;
    // XSUS
    if (ml[tL].max_size_cls != tree[root].edges && tree[root].nodes > 2 && tree[root].edges > 1)
      x[nodes - tL - 1] += pow(tree[root].edges, 2.) / pow(nodes, 2.);
    // ntrees
    if (dc <= 0) nt[nodes - tL - 1]++;
    // NEC
    if (tree[root].nodes > 2 && tree[root].edges > 1) nec[nodes - tL - 1]++;
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      vertex_entropy(tree, ml, *roo, nodes, Sh, Dc, x, nec, nt);
    }
  }
}

void level_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, std::string &root, int &nodes, std::vector<double> &sv, std::vector<double> &xx, std::vector<double> &orp) {
  int M = 1;
  double xm, x;
  sum_vertices(tree, root, M);
  for (std::map<int, level_properties>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    // Vertical entropy
    sv[nodes - ll->first - 1] -= static_cast<double>(ll->second.total_cls) * log(static_cast<double>(ll->second.total_cls) / M) / nodes;
    // XMSUS
    xm = 0, x = 0;
    for (std::map<int, int>::iterator v = ll->second.sizes_cls.begin(); v != ll->second.sizes_cls.end(); ++v) {
      xm += pow(v->first, 2.) * v->second;
      x += v->first * v->second;
    } 
    xx[nodes - ll->first - 1] += xm / x;
    // OrP
    orp[nodes - ll->first - 1] += static_cast<double>(ll->second.max_size_cls) / nodes;
  }
}