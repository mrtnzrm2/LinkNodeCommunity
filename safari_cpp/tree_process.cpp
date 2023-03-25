#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include <iterator>

struct vertex_properties {
  int level;
  int k;
  int m;
  double D;
  double height;
  std::set<int> node_pred;
  std::set<std::string> post_key;
};

struct tracer_properties {
  double D;
  int m;
  std::set<int> node_pred;
  std::set<std::string> post_keys;
};

struct level_properties {
  int size;
  double height;
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

bool search_key(std::map<int, tracer_properties> &a, int &key) {
  for (std::map<int, tracer_properties>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first == key) return true;
  }
  return false;
}

bool search_key(std::vector<std::pair<int, tracer_properties> > &a, int &key) {
  for (std::vector<std::pair<int, tracer_properties> >::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first == key) return true;
  }
  return false;
}

bool search_key(std::vector<std::pair<int, tracer_properties> > &a, const int &key) {
  for (std::vector<std::pair<int, tracer_properties> >::iterator it = a.begin(); it != a.end(); ++it) {
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

bool search_key(std::map<int, std::set<std::string> > &a, int &key) {
  for (std::map<int, std::set<std::string> >::iterator it = a.begin(); it != a.end(); ++it) {
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

std::set<int> unique_with_pred(std::vector<int> &a, std::set<int> & pred) {
  std::set<int> s;
  for (std::set<int>::iterator v = pred.begin(); v != pred.end(); ++v) {
    s.insert(a[*v]);
  }
  return s;
}

int number_of_nodes(std::set<int> &nodes, std::vector<int> &source, std::vector<int> &target) {
  std::set<int> N;
  for (std::set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    N.insert(source[*it]);
    N.insert(target[*it]);
  }
  return N.size();
}

double Dc(std::set<int> &nodes, std::vector<int> &source, std::vector<int> &target) {
  double m = nodes.size();
  double n = number_of_nodes(nodes, source, target);
  if (m > 1 && n  > 2)
    return (m - n + 1.) / pow(n - 1., 2.);
  else
    return 0.;
}

void Z2dict_short(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties> &tree,
  const std::string key_pred, std::set<int> nodes_pred, std::vector<double> &H, std::vector<int> &source, std::vector<int> &target, const int L, const int tL, const int &nodes
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
        Z2dict_short(A, tree, key_pred, nodes_com, H, source, target, next_L, tL, nodes);
      }
      else if (nodes_com.size() < nodes_pred.size()) {
        if (!search_key(tree, key_pred)) {
          tree[key_pred].level = tL;
          tree[key_pred].k = L;
          tree[key_pred].node_pred = nodes_pred;
          tree[key_pred].m = nodes_pred.size();
          tree[key_pred].D = Dc(nodes_pred, source, target); // * nodes_pred.size() / nodes;
          tree[key_pred].height = H[nodes - L];
          tree[key_pred].post_key.insert(key_pred + key);
        }
        else {
          tree[key_pred].post_key.insert(key_pred + key);
        }
        Z2dict_short(A, tree, key_pred + key, nodes_com, H, source, target, next_L, next_tL, nodes);
      }
    }
  } else {
    if (!search_key(tree, key_pred)) {
      tree[key_pred].level = tL;
      tree[key_pred].k = L;
      tree[key_pred].node_pred = nodes_pred;
      tree[key_pred].m = nodes_pred.size();
      tree[key_pred].D = Dc(nodes_pred, source, target);
      tree[key_pred].height = H[nodes - L];
      tree[key_pred].post_key.insert("END");
    }
  }
}

bool cmp(std::pair<int, tracer_properties > &a, std::pair<int, tracer_properties > &b) {
  return a.first > b.first;
}

std::vector<std::pair<int, tracer_properties > > sort_map(std::map<int, tracer_properties> &M) {
  std::vector<std::pair<int, tracer_properties > > A;
  for (auto& it : M) {
    A.push_back(it);
  }
  sort(A.begin(), A.end(), cmp);
  return A;
}

std::vector<std::pair<int, tracer_properties> > tract_tracing(std::map<std::string, vertex_properties> &tree, const int &nodes) {
  std::map<int, tracer_properties > tracer;
  for (std::map<std::string, vertex_properties>::iterator leaf=tree.begin(); leaf != tree.end(); ++leaf) {
    if (!search_key(tracer, leaf->second.k) && leaf->second.D > 0) {
      tracer[leaf->second.k].node_pred = leaf->second.node_pred;
      tracer[leaf->second.k].post_keys = leaf->second.post_key;
      tracer[leaf->second.k].D = leaf->second.D;
      tracer[leaf->second.k].m = leaf->second.m;
    }
  }
  for (int i=1 ; i < nodes; i ++) {
    if (!search_key(tracer, i)) {
      tracer[i].D = 0;
      tracer[i].post_keys.insert("END");
    }
  }
  std::vector<std::pair<int, tracer_properties> > S = sort_map(tracer);
  return S;
}

void sum_vertices(std::map<std::string, vertex_properties> &tree, const std::string &root, int &t) {
  for (std::set<std::string>::iterator s = tree[root].post_key.begin(); s != tree[root].post_key.end(); ++s) {
    if ((*s).compare("END") == 0) continue;
    t++;
    sum_vertices(tree, *s, t);
  }
}
  // double last_m = 1;
  // {0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0, 0 ,0}, // 1
  // {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, // 2
  // {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1}, // 3
  // {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3}, // 4
  // {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 3, 4, 3, 3}, // 5
  // {0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 5, 4, 4}, // 6
  // {0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 4, 4}, //7
  // {0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 5, 5}, // 8
  // {0, 0, 1, 1, 2, 3, 3, 3, 4, 3, 5, 6, 7, 8, 6, 6}, // 9
  // {0, 0, 1, 1, 2, 3, 3, 4, 5, 4, 6, 7, 8, 9, 7, 7}, // 10
  // {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 8, 8}, // 11
  // {0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 9, 9}, // 12
  // {0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 9}, // 13
  // {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 10}, // 14
  // {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, // 15
  // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} // 16
void D(std::vector<std::pair<int, tracer_properties> > &tracer, std::map<std::string, vertex_properties> &tree, std::vector<double> &d, const int &nodes) {
  std::set<int> used_nodes;
  std::vector<int> used_nodes2;
  std::set<int> compared;
  for (std::vector<std::pair<int, tracer_properties> >::iterator it = tracer.begin(); it != tracer.end(); ++it) {
    if (search_key(it->second.post_keys, "END")) {
      d[it->first - 1] += d[it->first];
    } else {
      compared = intersection(used_nodes, it->second.node_pred);
      if (compared.size() > 0) {
        auto it1 = next(it->second.post_keys.begin(), 0);
        auto it2 = next(it->second.post_keys.begin(), 1);
        d[it->first - 1] = d[it->first]  + (it->second.D - tree[*it1].D - tree[*it2].D) * it->second.m / nodes;
      } else {
        d[it->first - 1] = d[it->first] + it->second.D * it->second.m / nodes;
      }
      std::set_union(used_nodes.begin(), used_nodes.end(), it->second.node_pred.begin(), it->second.node_pred.end(), std::back_inserter(used_nodes2));
      used_nodes.clear();
      for (auto u : used_nodes2)
        used_nodes.insert(u);
    }
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

std::vector<std::pair<int, tracer_properties > > Z2dict(std::vector<std::vector<int> > &A,  std::map<std::string, vertex_properties> &tree, std::vector<double> &H, std::vector<int> &source, std::vector<int> &target, std::string &type) {
  const int N = A.size();
  std::set<int> nodes;
  for (int i=0; i < A.size(); i++) {
    nodes.insert(A[A.size() - 1][i]);
  }
  const int L = 1, tL = 0;
  const std::string root = "L00";
  if (type.compare("short") == 0) Z2dict_short(A, tree, root, nodes, H, source, target, L, tL, N);
  else {
    throw std::runtime_error("\nOnly types: short\n");
  }
  std::vector<std::pair<int, tracer_properties> > tracer = tract_tracing(tree, N);
  return tracer;
}

int main() {
  std::string root = "L00";
  std::string t_short = "short";

  std::vector<std::vector<int> > a{
    {0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0, 0 ,0}, // 1
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, // 2
    {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1}, // 3
    {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3}, // 4
    {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 3, 4, 3, 3}, // 5
    {0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 5, 4, 4}, // 6
    {0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 4, 4}, //7
    {0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 5, 5}, // 8
    {0, 0, 1, 1, 2, 3, 3, 3, 4, 3, 5, 6, 7, 8, 6, 6}, // 9
    {0, 0, 1, 1, 2, 3, 3, 4, 5, 4, 6, 7, 8, 9, 7, 7}, // 10
    {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 8, 8}, // 11
    {0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 9, 9}, // 12
    {0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 9}, // 13
    {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 10}, // 14
    {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, // 15
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} // 16
  };

  std::vector<std::vector<int> > b{
    {1, 1, 1, 1, 1},
    {1, 1, 2, 2, 2},
    {1, 1, 2, 2, 3},
    {1, 2, 3, 3, 4},
    {1, 2, 3, 4, 5}
  };

  int nodes = 16;

  std::map<std::string, vertex_properties> tree;
  std::map<int, level_properties> chain;

  std::vector<int> out_a{0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6};
  std::vector<int> in_a{2, 3, 4, 5, 6, 0, 2, 3, 0, 3, 1, 5, 6, 4, 4, 5};

  std::vector<int> out_b{0, 0, 0, 3, 4};
  std::vector<int> in_b{3, 1, 3, 1, 2};

  std::vector<double> h_a{
    0, 0.    ,     0.  ,       0.    ,     0.046875 ,  0.046875  , 0.046875,
    0.07465278, 0.11631944, 0.11631944, 0.19444444, 0.31944444 ,0.42881944,
    0.47222222 ,0.234375 ,  0.27777778
  };

  std::vector<double> h_b{ 0.,  0.046875 ,  0.046875  , 0.046875, 0.07465278};

  std::vector<std::pair<int, tracer_properties > > tracer = Z2dict(a, tree, h_a, out_a, in_a, t_short);

  std::vector<double> d(nodes, 0.);
  D(tracer, tree, d, nodes);

  for (auto it : d) {
    std::cout << it << " ";
  }

  std::cout << "\n";

  for (std::map<std::string, vertex_properties>::iterator it = tree.begin(); it != tree.end(); ++it) {
    std::cout << it->first << "\t\t\t" << it->second.k << "\t" << it->second.D << std::endl;
    // for (std::set<std::string>::iterator ii = it->second.post_key.begin(); ii != it->second.post_key.end(); ii++) {
    //   std::cout << *ii << "\t\t";
    // }
    // std::cout << "\n\n";
  }

  for (auto& it : tracer) {
    std::cout << it.first << "\t" << it.second.D << std::endl;
    for (std::set<std::string>::iterator ip=it.second.post_keys.begin(); ip != it.second.post_keys.end(); ++ip)
      std::cout << *ip << " ";
    std::cout << "\n";
  }

  return 0;
}

