#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include <iterator>

struct vertex_properties_bene {
  int level;
  int k;
  int m;
  double D;
  double height;
  std::set<int> node_pred;
  std::set<std::string> post_key;
};

struct tracer_properties {
  std::vector<std::string> neighbors;
};

std::set<int> intersection_bene(std::set<int> &a, std::set<int> &b) {
  std::set<int> intersect;
  std::set_intersection(
    a.begin(), a.end(), b.begin(), b.end(),
    std::inserter(intersect, intersect.begin())
  );
  return intersect;
}

std::set<int> where_bene(std::vector<int> &v, std::set<int> &pred,const int &com) {
  std::set<int> where_bene_com;
  for (std::set<int>::iterator it = pred.begin(); it != pred.end(); ++it) {
    if (v[*it] == com) {
      where_bene_com.insert(*it);
    }
  }
  return where_bene_com;
}

bool search_key_bene(std::map<std::string, vertex_properties_bene> &a, const std::string &key) {
  for (std::map<std::string, vertex_properties_bene>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->first.compare(key) == 0) return true;
  }
  return false;
}

std::set<int> unique_with_pred_bene(std::vector<int> &a, std::set<int> & pred) {
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

void Z2dict_long(
  std::vector<std::vector<int> > &A, std::map<std::string, vertex_properties_bene> &tree,
  const std::string key_pred, std::set<int> nodes_pred, std::vector<double> &H, std::vector<int> &source, std::vector<int> &target, const int L, const int tL, const int &nodes
) {
  if (L < A.size() && nodes_pred.size() > 1) {
    const int next_L = L + 1, next_tL = tL + 1;
    std::set<int> coms = unique_with_pred_bene(A[L], nodes_pred);
    std::set<int> nodes_com;
    std::set<int> compare;
    for (std::set<int>::iterator com = coms.begin(); com != coms.end(); ++com) {
      const std::string key = "L" + std::to_string(next_tL) + std::to_string(*com);
      nodes_com = where_bene(A[L], nodes_pred, *com);
      compare = intersection_bene(nodes_com, nodes_pred);
      if (compare.size() == 0) continue;
      if (nodes_com.size() <= nodes_pred.size()) {
        if (!search_key_bene(tree, key_pred)) {
          tree[key_pred].level = tL;
          tree[key_pred].k = L;
          tree[key_pred].m = nodes_pred.size();
          tree[key_pred].D = Dc(nodes_pred, source, target) * nodes_pred.size() / nodes;
          tree[key_pred].height = H[nodes - L];
          tree[key_pred].post_key.insert(key_pred + key);
        }
        else {
          tree[key_pred].post_key.insert(key_pred + key);
        }
        Z2dict_long(A, tree, key_pred + key, nodes_com, H, source, target, next_L, next_tL, nodes);
      }
    }
  } else {
    if (!search_key_bene(tree, key_pred)) {
      tree[key_pred].level = tL;
      tree[key_pred].k = L;
      tree[key_pred].m = nodes_pred.size();
      tree[key_pred].D = Dc(nodes_pred, source, target);
      tree[key_pred].height = H[nodes - L];
      tree[key_pred].post_key.insert("END");
    }
  }
}

std::map<int, tracer_properties > tract_tracing(std::map<std::string, vertex_properties_bene> &tree, const int &nodes) {
  std::map<int, tracer_properties > tracer;
  for (std::map<std::string, vertex_properties_bene>::iterator leaf=tree.begin(); leaf != tree.end(); ++leaf) {
    if (leaf->second.k == nodes) continue;
    tracer[leaf->second.k].neighbors.push_back(leaf->first);
  }
  return tracer;
}

void BUONO(std::map<int, tracer_properties > &tracer, std::map<std::string, vertex_properties_bene> &tree, std::vector<std::vector<double> > &BENE, int &alpha, double &beta, const int &nodes) {
  /// 0: NEC; 1: MU; 2: D; 3 : ntrees; 4: X; 5: M ///
  int lci, lcj, K = nodes - 1;
  double d;
  std::vector<std::vector<int> > resevoir(nodes - 1);
  for (std::map<int, tracer_properties>::iterator tr = tracer.begin(); tr != tracer.end(); ++tr) {
    for (std::vector<std::string>::iterator keys=tr->second.neighbors.begin(); keys != tr->second.neighbors.end(); ++keys) {
      resevoir[nodes - tr->first - 1].push_back(tree[*keys].m);
      BENE[2][nodes - tr->first - 1] += tree[*keys].D;
      if (tree[*keys].D == 0)
        BENE[3][nodes - tr->first - 1]++;
      else
        BENE[0][nodes - tr->first - 1]++;
    }
  }
  for (std::vector<std::vector<int> >::iterator re = resevoir.begin(); re != resevoir.end(); ++re) {
    sort(re->begin(), re->end(), std::greater<int>());
    BENE[5][nodes - K - 1] = static_cast<double>(*std::next(re->begin(), 0)) / nodes;
    for (auto r : *re) {
      if (r <= 1) continue;
      if (r != *std::next(re->begin(), 0))
        BENE[4][nodes - K - 1] += pow(r, 2.);
    }
    BENE[4][nodes - K - 1] /= nodes;
    if (K >= alpha) {
      for (int i=0; i < alpha - 1; i++) {
        for (int j=i + 1; j < alpha; j++) {
          if (i < re->size())
            lci = *std::next(re->begin(), i);
          else lci = 1;
          if (j < re->size())
            lcj = *std::next(re->begin(), j);
          else lcj = 1;
          d = lcj / (lci * 1.) ;
          if (d > beta)
            BENE[1][nodes - K - 1] += d * (lcj + lci) / (2. * nodes);
          else
            BENE[1][nodes- K - 1] -= d * (lcj + lci) / (2. * nodes);
        }
      }
      BENE[1][nodes - K - 1] /= 0.5 * alpha * (alpha - 1);
    } else if (K >= 2) {
      for (int i=0; i < K - 1; i++) {
        for (int j=i + 1; j < K; j++) {
          if (i < re->size())
            lci = *std::next(re->begin(), i);
          else lci = 1;
          if (j < re->size())
            lcj = *std::next(re->begin(), j);
          else lcj = 1;
          d = lcj / (lci * 1.);
          if (d > beta)
            BENE[1][nodes - K - 1] += d * (lci + lcj) / (2. * nodes);
          else
            BENE[1][nodes - K - 1] -= d * (lci + lcj) / (2. * nodes);
        }
      }
      BENE[1][nodes - K - 1] /= 0.5 * K  * (K - 1);
    } else {
      BENE[1][nodes - K - 1] = 0.;
    }
    K--;
  }
}

void  Z2dict_bene(std::vector<std::vector<int> > &A,  std::map<std::string, vertex_properties_bene> &tree, std::map<int, tracer_properties > &tracer, std::vector<double> &H, std::vector<int> &source, std::vector<int> &target, std::string &type) {
  const int N = A.size();
  std::set<int> nodes;
  for (int i=0; i < A.size(); i++) {
    nodes.insert(A[A.size() - 1][i]);
  }
  const int L = 1, tL = 0;
  const std::string root = "L00";
  if (type.compare("long") == 0) Z2dict_long(A, tree, root, nodes, H, source, target, L, tL, N);
  else {
    throw std::runtime_error("\nOnly types: long\n");
  }
  tracer = tract_tracing(tree, N);
}