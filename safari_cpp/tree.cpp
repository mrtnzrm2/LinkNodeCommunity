#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>

struct vertex_properties {
  int level;
  std::vector<std::string> post_key{"ZZZZZZZZZZZZZZZ", "ZZZZZZZZZZZZZZZ"};
};

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

std::map<int, double > link_communitiy_Dc(
  std::vector<int> labels, int& leaves,
  std::vector<int>& source, std::vector<int>& target
) {
  int edg;
  std::vector<int> nodes_src;
  std::vector<int> nodes_tgt;
  std::vector<int> lc_nodes;
  unique_2(labels);
  std::map<int, double> Dc;
  for (int i = 0; i < labels.size(); i++) {
    edg = 0;
    for (int j = 0; j < leaves; j++) {
      if ( labels[j] == labels[i]) {
        nodes_src.push_back(source[j]);
        nodes_tgt.push_back(target[j]);
        edg++;
      }
    }
    lc_nodes = nodes_src;
    lc_nodes.insert(
      lc_nodes.end(), nodes_tgt.begin(), nodes_tgt.end()
    );
    unique_2(lc_nodes);
    int N = lc_nodes.size();
    double dc = static_cast<double>(edg - N + 1) / static_cast<double>(pow(N - 1, 2.0));
    Dc.insert(std::pair<int, double>(labels[i], dc));
    nodes_src.clear();
    nodes_tgt.clear();
  }
  return Dc;
}

struct node_merde {
  double dc;
  std::set<int> pos;
};

std::map<int, node_merde > unique_tree_Dc(std::vector<std::vector<int> > &A, std::map<int, double> &Dc, int &l) {
  std::map<int, node_merde > coms;
  for (int i=0; i < A[l].size(); i++) {
    coms[A[l][i]].dc = Dc[A[l][i]];
    coms[A[l][i]].pos.insert(i);
  }
  return coms;
}

void Z2dict_short_Dc(
  std::vector<std::vector<int> > &A, std::vector<int> &src, std::vector<int> &tar, std::map<std::string, vertex_properties> &tree,
  std::string key_pred, std::set<int> nodes_pred, int L, int tL
) {
  if (L < A.size()) {
    int next_L = L + 1, next_tL = tL + 1, size = A[L].size();
    std::map<int, double> Dcs = link_communitiy_Dc(A[L], size, src, tar);
    std::map<int, node_merde > coms = unique_tree_Dc(A, Dcs, L);
    std::string key;
    std::set<int> nodes_com;
    std::set<int> compare;
    for (std::map<int, node_merde >::iterator cc = coms.begin(); cc != coms.end(); ++cc) {
      key = key_pred + "L" + std::to_string(tL + 1) + std::to_string(cc->first);
      nodes_com = cc->second.pos;
      if (cc->second.dc > 0) {
        compare = intersection(nodes_com, nodes_pred);
        if (compare.size() == 0) continue;
        if (nodes_com.size() == nodes_pred.size()) {
          Z2dict_short_Dc(A, src, tar, tree, key_pred, nodes_com, next_L, tL);
        }
        else if (nodes_com.size() < nodes_pred.size()) {
          if (!search_key(tree, key_pred)) {
            vertex_properties item;
            item.level = tL;
            item.post_key.push_back(key);
            tree.insert(std::pair<std::string, vertex_properties>(key_pred, item));
          }
          else {
            tree[key_pred].post_key.push_back(key);
          }
          Z2dict_short_Dc(A, src, tar, tree, key, nodes_com, next_L, next_tL);
        }
      } else {
        if (!search_key(tree, key_pred)) {
          vertex_properties item;
          item.level = tL;
          item.post_key.push_back(key);
          tree.insert(std::pair<std::string, vertex_properties>(key_pred, item));
        }
        if (!search_key(tree[key_pred].post_key ,  key)) {
          tree[key_pred].post_key.push_back(key);
        }
        if (!search_key(tree, key)) {
          vertex_properties item;
          item.level = tL + 1;
          item.post_key.push_back("END");
          tree.insert(std::pair<std::string, vertex_properties>(key, item));
        }
      }
    }
  } 
}

void Z2dict_short(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  std::string key_pred, std::set<int> nodes_pred, int L, int tL
) {
  if (L < A.size()) {
    int next_L = L + 1, next_tL = tL + 1;
    std::vector<int> coms = A[L];
    unique_2(coms);
    std::string key;
    std::set<int> nodes_com;
    std::set<int> compare;
    for (int i=0; i < coms.size(); i++) {
      key = "L" + std::to_string(next_tL) + std::to_string(coms[i]);
      nodes_com = where(A[L], coms[i]);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() == 0) continue;
      if (nodes_com.size() == nodes_pred.size()) {
        Z2dict_short(A, tree, key_pred, nodes_com, next_L, tL);
      }
      else if (nodes_com.size() < nodes_pred.size()) {
        if (!search_key(tree, key_pred)) {
          tree[key_pred].level = tL;
          tree[key_pred].post_key[0] = key_pred + key;
        }
        else {
          tree[key_pred].post_key[1] = key_pred + key;
        }
        Z2dict_short(A, tree, key_pred + key, nodes_com, next_L, next_tL);
      }
    }
  } else {
    std::vector<std::string> end_string{"END"};
    tree[key_pred].level = tL;
    tree[key_pred].post_key = end_string;
  }
}

void Z2dict_long(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  std::string key_pred, std::set<int> nodes_pred, int L, int tL
) {
  if (L < A.size()) {
    int next_L = L + 1, next_tL = tL + 1;
    std::vector<int> coms = A[L];
    unique_2(coms);
    std::string key;
    std::set<int> nodes_com;
    std::set<int> compare;
    for (int i=0; i < coms.size(); i++) {
      key = "L" + std::to_string(next_tL) + std::to_string(coms[i]);
      nodes_com = where(A[L], coms[i]);
      compare = intersection(nodes_com, nodes_pred);
      if (compare.size() > 0) {
        if (!search_key(tree, key_pred)) {
          tree[key_pred].level = tL;
          tree[key_pred].post_key[0] = key_pred + key;
        }
        else {
          tree[key_pred].post_key[1] = key_pred + key;
        }
        Z2dict_long(A,tree, key_pred + key, nodes_com, next_L, next_tL);
      }
    }
  } else {
    std::vector<std::string> end_string{"END"};
    tree[key_pred].level = tL;
    tree[key_pred].post_key = end_string;
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
  if (type.compare("short") == 0) Z2dict_short(A, tree, root, nodes, L, tL);
  else if (type.compare("long") == 0) Z2dict_long(A, tree, root, nodes, L, tL);
  else if (type.compare("short_DC") == 0) Z2dict_short_Dc(A, src, tar, tree, root, nodes, L, tL);
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

void level_information(std::map<std::string, vertex_properties> &tree, std::string &root, std::map<int, int> &ml) {
  for (std::map<std::string, vertex_properties>::iterator t = tree.begin(); t != tree.end(); ++t) {
    if (!search_key(ml, t->second.level)) {
      ml.insert(std::pair<int, int>(t->second.level, 1));
    } else {
      ml[t->second.level]++;
    }
  }
}

void vertex_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, int> ml, std::string &root, int &nodes, std::vector<double> &Sh) {
  double Mul;
  int nextL = tree[root].level + 1, tL = tree[root].level;
  if (!search_key(tree[root].post_key, "END")) {
    // Horizontal Entropy
    Mul = tree[root].post_key.size();
    Sh[nodes - tL - 1] -= Mul * log(Mul / static_cast<double>(ml[nextL])) / nodes;
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      vertex_entropy(tree, ml, *roo, nodes, Sh);
    }
  }
}

void level_entropy(std::map<std::string, vertex_properties> &tree, std::map<int, int> &ml, std::string &root, int &nodes, std::vector<double> &sv) {
  int M = 1;
  double xm, x;
  sum_vertices(tree, root, M);
  for (std::map<int, int>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    // Vertical entropy
    sv[nodes - ll->first - 1] -= static_cast<double>(ll->second) * log(static_cast<double>(ll->second) / M) / nodes;
  }
}

bool find_pocoyo(std::map<std::string, vertex_properties> &v, std::string &root,  const char* key) {
  for (std::vector<std::string>::iterator roo = v[root].post_key.begin(); roo != v[root].post_key.end(); ++roo) {
    if (search_key(v[*roo].post_key, key)) return true;
  }
  return false;
}

void vertex_entropy_H(std::map<std::string, vertex_properties> &tree, std::map<int, int> ml, std::string &root, int &nodes, std::vector<double> &H, std::vector<double> &Sh) {
  bool pocoyo = false;
  double Mul;
  int nextL = tree[root].level + 1, tL = tree[root].level;
  if (!search_key(tree[root].post_key, "END")) {
    Mul = tree[root].post_key.size();
    pocoyo = find_pocoyo(tree, root, "END");
    if (pocoyo)
      Sh[nodes - tL - 1] -= Mul * H[tL] * log(Mul / static_cast<double>(ml[nextL])) / nodes;
    else
      Sh[nodes - tL - 1] -= Mul * (H[tL] - H[tL - 1]) * log(Mul / static_cast<double>(ml[nextL])) / nodes;
    for (std::vector<std::string>::iterator roo = tree[root].post_key.begin(); roo != tree[root].post_key.end(); ++roo) {
      vertex_entropy_H(tree, ml, *roo, nodes, H, Sh);
    }
  }
}

void level_entrop_H(std::map<std::string, vertex_properties> &tree, std::map<int, int> &ml, std::string &root, int &nodes, std::vector<double> &H, std::vector<double> &sv) {
  int M = 1;
  sum_vertices(tree, root, M);
  std::cout << "Number of vertices in the tree: " << M << "\n";
  for (std::map<int, int>::iterator ll = ml.begin(); ll != ml.end(); ++ll) {
    // Vertical entropy
    if (ll->first > 0)
      sv[nodes - ll->first - 1] -= static_cast<double>(ll->second) * (H[ll->first] - H[ll->first - 1]) * log(static_cast<double>(ll->second) / M) / nodes;
    else
      sv[nodes - ll->first - 1] -= static_cast<double>(ll->second) * H[ll->first] * log(static_cast<double>(ll->second) / M) / nodes;
  }
}

int main() {
  std::string root = "L00";
  std::string t_short = "short";
  std::string t_long = "long";
  std::string t_short_DC = "short_DC";

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

  int nodes = 16;
  std::vector<double> Sh(nodes, 0.);


  std::vector<double> Sv(nodes, 0.);

  std::map<int, int> chain;

  std::vector<int> out_a{0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6};
  std::vector<int> in_a{2, 3, 4, 5, 6, 0, 2, 3, 0, 3, 1, 5, 6, 4, 4, 5};

  std::vector<int> out_b{0, 0, 0, 3, 4};
  std::vector<int> in_b{3, 1, 3, 1, 2};

  std::map<std::string, vertex_properties> tree_long = Z2dict(a, out_a, in_a, t_short);
  level_information(tree_long, root, chain);

  // for (std::map<int, level_properties>::iterator v = chain.begin(); v != chain.end(); ++v){
  //   std::cout << v->first << "\n";
  //   for (std::map<int, int>::iterator vv = v->second.sizes_cls.begin(); vv != v->second.sizes_cls.end(); ++vv)
  //     std::cout << vv->first << "\t" << vv->second << "\n";
  // }

  // for (std::map<std::string, vertex_properties>::iterator it = tree_long.begin(); it != tree_long.end(); ++it) {
  //   std::cout << it->first << "\t\t\t" << it->second.level << std::endl;

  //   for (std::vector<std::string>::iterator ii = it->second.post_key.begin(); ii != it->second.post_key.end(); ii++) {
  //     std::cout << *ii << "\t\t";
  //   }
  //   std::cout << "\n\n";
  // }

  vertex_entropy(tree_long, chain, root, nodes, Sh);
  level_entropy(tree_long, chain, root, nodes, Sv);

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