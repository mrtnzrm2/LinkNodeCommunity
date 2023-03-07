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
void unique(std::vector<T> &v) {
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
    unique(coms);
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
    unique(coms);
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
    orp[nodes - ll->first - 1] += ll->second.max_size_cls / nodes;
  }
}

void D(std::map<std::string, vertex_properties> &tree, std::string &root, std::vector<double> &Dc, int &nodes) {
  double dc;
  if (!search_key(tree[root].post_key, "END")) {
    dc = (tree[root].edges - tree[root].nodes + 1) / (pow(tree[root].nodes - 1, 2.));
    Dc[tree[root].level] += dc * tree[root].edges / nodes;
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      D(tree, *roo, Dc, nodes);
    }
  }
}

void X(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, std::string &root, std::vector<double> &x, int &nodes) {
  if (!search_key(tree[root].post_key, "END")) {
    if (ml[tree[root].level].max_size_cls != tree[root].edges && tree[root].nodes > 2 && tree[root].edges > 1)
      x[tree[root].level] += pow(tree[root].edges, 2.) / pow(nodes, 2.);
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      X(tree, ml, *roo, x, nodes);
    }
  }
}

void SH(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> ml, std::string &root, std::vector<double> &Sh, int &nodes) {
  double Mul;
  int nextL = tree[root].level + 1;
  if (!search_key(tree[root].post_key, "END")) {
    Mul = tree[root].post_key.size();
    Sh[tree[root].level] -= Mul * log(Mul / static_cast<double>(ml[nextL].total_cls)) / nodes;
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      SH(tree, ml, *roo, Sh, nodes);
    }
  }
}

void  SV(std::map<std::string, vertex_properties> &tree, std::map<int, level_properties> &ml, std::string &root, std::vector<double> &sv, int &nodes) {
  int M = 1;
  sum_vertices(tree, root, M);
  for (std::map<int, level_properties>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    sv[ll->first] -= static_cast<double>(ll->second.total_cls) * log(static_cast<double>(ll->second.total_cls) / M) / nodes;
  }
}

void Xm(std::map<int, level_properties> &ml, std::vector<double> &xx) {
  double xm, x;
  for (std::map<int, level_properties>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    xm = 0, x = 0;
    for (std::map<int, int>::iterator v = ll->second.sizes_cls.begin(); v != ll->second.sizes_cls.end(); ++v) {
      xm += pow(v->first, 2.) * v->second;
      x += v->first * v->second;
    } 
    xx[ll->first] += xm / x;
  }

}

int main() {
  std::string root = "L00";
  std::string t_short = "short";
  std::string t_long = "long";

  int nodes = 16;
  std::vector<double> Sh(nodes, 0.);
  std::vector<double> Dav(nodes, 0.);
  std::vector<double> X_sus(nodes, 0.);
  std::vector<int> NEC(nodes, 0);
  std::vector<int> ntrees(nodes, 0);


  std::vector<double> Sv(nodes, 0.);
  std::vector<double> Xm_sus(nodes, 0.);
  std::vector<double> OrP(nodes, 0.);

  std::map<int, level_properties> chain;


  std::vector<std::vector<int> > a{
    {0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0, 0 ,0},
    {0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0 ,1, 0, 0, 0 ,0 ,0},
    {0 ,0 ,1 ,1 ,1, 0 ,0 ,0, 0 ,0 ,2 ,1, 1 ,1,1 ,1},
    {0, 0 ,1, 1 ,1 ,0, 0 ,0 ,0 ,0, 2, 3 ,3 ,3, 3, 3},
    {0 ,0 ,1, 1, 1, 0, 0, 0, 0 ,0, 2 ,3 ,3, 4, 3, 3},
    {0 ,0 ,1, 1, 1, 2 ,2 ,2, 2, 2 ,3, 4 ,4 ,5,4, 4},
    {0 ,0 ,1 ,1 ,2, 3 ,3 ,3, 3 ,3 ,4 ,5, 5 ,6 ,5 ,5},
    {0 ,0, 1 ,1, 2, 3, 3 ,3, 3, 3 ,4 ,5 ,6, 7, 5 ,5},
    {0 ,0 ,1 ,1, 2, 3, 3 ,3, 4 ,3 ,5, 6, 7, 8, 6, 6},
    {0 ,1 ,2 ,2 ,3, 4, 4 ,4 ,5 ,4 ,6 ,7 ,8, 9, 7, 7},
    {0, 1  ,2 , 2  ,3 , 4,  4,  5 , 6,  5, 7,  8,  9 ,10 , 8 , 8},
    {0, 1 , 2 , 2 ,3  ,4 , 4,  5,  6 , 7 , 8,  9 ,10, 11 , 9,  9},
    {0 ,1  ,2 , 3 , 4,  5 , 5 , 6 , 7,  8 , 9,10 ,11, 12, 10, 10},
    {0 ,1 , 2 , 3 , 4  ,5  ,5 , 6,  7  ,8 , 9, 10, 11, 12 ,13, 10},
    {0 ,1 , 2  ,3  ,4 , 5,  6 , 7 , 8 , 9 ,10 ,11 ,12, 13, 14 ,11},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  };

  std::vector<std::vector<int> > b{
    {1, 1, 1, 1, 1},
    {1, 1, 2, 2, 2},
    {1, 1, 2, 2, 3},
    {1, 2, 3, 3, 4},
    {1, 2, 3, 4, 5}
  };

  std::vector<int> out_a{0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6};
  std::vector<int> in_a{2, 3, 4, 5, 6, 0, 2, 3, 0, 3, 1, 5, 6, 4, 4, 5};

  std::vector<int> out_b{0, 0, 0, 3, 4};
  std::vector<int> in_b{3, 1, 3, 1, 2};

  // std::map<std::string, vertex_properties> tree_short = Z2dict(a, out, in, t_shot);
  std::map<std::string, vertex_properties> tree_long = Z2dict(a, out_a, in_a, t_long);
  level_information(tree_long, root, chain);

  // for (std::map<int, level_properties>::iterator v = chain.begin(); v != chain.end(); ++v){
  //   std::cout << v->first << "\n";
  //   for (std::map<int, int>::iterator vv = v->second.sizes_cls.begin(); vv != v->second.sizes_cls.end(); ++vv)
  //     std::cout << vv->first << "\t" << vv->second << "\n";
  // }

  // for (std::map<std::string, vertex_properties>::iterator it = tree_long.begin(); it != tree_long.end(); ++it) {
  //   std::cout << it->first << "\t\t\t" << it->second.level << "\t" << it->second.edges << "\t" << it->second.nodes << std::endl;

  //   for (std::vector<std::string>::iterator ii = it->second.post_key.begin(); ii != it->second.post_key.end(); ii++) {
  //     std::cout << *ii << "\t\t";
  //   }
  //   std::cout << "\n\n";
  // }

  // SV(tree_long, chain, root, Sv, nodes);
  // Xm(chain, Xm_sus);

  // SH(tree_long, chain, root, Sh, nodes);
  // D(tree_long, root, Dav, nodes);
  // X(tree_long, chain, root, X_sus, nodes);

  vertex_entropy(tree_long, chain, root, nodes, Sh, Dav, X_sus, NEC, ntrees);
  level_entropy(tree_long, chain, root, nodes, Sv, Xm_sus, OrP);

  for (auto d : Dav) {
    std::cout << d << " ";
  }
  std::cout << "\n";

  // for (auto d : X_sus) {
  //   std::cout << d << " ";
  // }
  // std::cout << "\n";

  // for (auto d : Xm_sus) {
  //   std::cout << d << " ";
  // }
  // std::cout << "\n";

  // for (auto d : NEC) {
  //   std::cout << d << " ";
  // }
  // std::cout << "\n";
  
  // for (auto d : ntrees) {
  //   std::cout << d << " ";
  // }
  // std::cout << "\n";

  // for (auto d : OrP) {
  //   std::cout << d << " ";
  // }
  // std::cout << "\n\n";

  double total_sv = 0., total_sh = 0.;
  for (auto v : Sv) {
    std::cout << v << " ";
    total_sv += v;
  }
  std::cout << "\n" << total_sv << "\n";
  for (auto v : Sh) {
    std::cout << v << " ";
    total_sh += v;
  }
  std::cout << "\n" << total_sh << "\n";

  return 0;
}