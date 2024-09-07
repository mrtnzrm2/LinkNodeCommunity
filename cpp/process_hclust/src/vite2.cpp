#include <iostream>
#include <algorithm>
#include <numeric> 
#include <map>
#include <vector>
#include <array>
#include <set>

struct link_groups {
  std::vector<std::array<int, 2>> edgelist;
  std::set<int> link_nodes;
};

std::map<int, link_groups> group_linkcommunities(
  int* labels, std::vector<int>& source, std::vector<int>& target,
  int& n) {
    
    int u, v;
    std::map<int, link_groups> lgroups;
    std::array<int, 2> edges_nodes;
    std::set<int> link_nodes;

    for (int i=0; i<n; i++) {
      u = source[i];
      v = target[i];

      if (u > v) {edges_nodes[0] = v; edges_nodes[1] = u;}
      else {edges_nodes[1] = v; edges_nodes[0] = u;}

      lgroups[labels[i]].edgelist.push_back(edges_nodes);
      lgroups[labels[i]].link_nodes.insert(u);
      lgroups[labels[i]].link_nodes.insert(v);

    }

    return lgroups;
}

double local_clustering_cefficient(int& u, std::vector<std::array<int, 2>> &edgelist) {

  std::set<int> u_neighbors;
  double lcc;
  int ml = edgelist.size(), link_neighbors=0;

  // Collect u's neighbors

  for (int e=0; e < ml; e++){
    if (edgelist[e][0] == u ) {
      u_neighbors.insert(edgelist[e][1]);
    }
    else if (edgelist[e][1] == u) {
      u_neighbors.insert(edgelist[e][0]);
    }
  }

  // Count number of links between u neighbors

  for (int e=0; e< ml; e++) {
    if (
      (u_neighbors.find(edgelist[e][0]) != u_neighbors.end()) &&
      (u_neighbors.find(edgelist[e][1]) != u_neighbors.end())
    )
    link_neighbors++;
  }
  if (u_neighbors.size() > 1)
    lcc = 1. * link_neighbors / (u_neighbors.size()*(u_neighbors.size() - 1));
  else lcc = 0;
  return lcc;
}

double global_clustering_coefficient(std::set<int> &link_nodes, std::vector<std::array<int, 2>> &edgelist) {
  int node;
  double gcc = 0.;

  for (std::set<int>::iterator u=link_nodes.begin(); u != link_nodes.end(); ++u) {
    node =  *u;
    gcc += local_clustering_cefficient(node, edgelist);
  }

  // std::cout << gcc << " ";

  return 1. * gcc / link_nodes.size();
}

double lc_mean_cc(std::map<int, link_groups> &groups) {

  double gcc = 0.;
  for (std::map<int, link_groups>::iterator u = groups.begin(); u!= groups.end(); ++u) {

    gcc += global_clustering_coefficient(u->second.link_nodes, u->second.edgelist);

  }

  return 1. * gcc / groups.size();

}
// int main(){
//   int labels[4] = {1, 1, 2, 2};

//   std::vector<int> s = {0, 1, 2, 3};
//   std::vector<int> t = {4, 3, 2, 1};

//   int n = 4, m;

//   std::map<int, std::vector<std::array<int, 2>>> lgroup;

//   lgroup = group_linkcommunities(labels, s, t, n);
//   std::cout << lgroup.size()<< "\n";
//   std::cout << lgroup.begin()->first<< "\n";
//   m = lgroup.begin()->second[0][1];
//   std::cout << m << "\n";
// }

