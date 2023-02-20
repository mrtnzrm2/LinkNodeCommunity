#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include<ctime> // time

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::vector<double> histogram_grid(
  std::vector<double> p,
  std::vector<double> q,
  double &h
) {
  double min_p, min_q, max_p, max_q, carrier;
  double MAX, MIN;
  // Sort the vector ----
  std::sort(p.begin(), p.end());
  std::sort(q.begin(), q.end());
  std::vector<double> grid;
  // Get min and max element ----
  min_p = p[0];
  max_p = p[p.size() - 1];
  min_q = q[0];
  max_q = q[q.size() - 1];
  if (min_p < min_q) MIN = min_p;
  else MIN = min_q;
  if (max_p < max_q) MAX = max_q;
  else MAX = max_p;
  // Define grid ----
  carrier = MIN - h;
  // Add boundaries ----
  while (carrier <= MAX) {
    grid.push_back(carrier);
    carrier += h;
  }
  grid.push_back(carrier);
  return grid;
}

std::vector<double> histogram(
  std::vector<double> &p, std::vector<double> &grid
) {
  std::vector<double> hp(grid.size() - 1, 0);
  for (int i = 0; i < p.size(); i++) {
    for (int j = 0; j < grid.size() - 1; j++) {
      if (p[i] >= grid[j] && p[i] < grid[j + 1])
        hp[j]++;
    }
  }
  return hp;
}

double sum_histogram(std::vector<double> &p) {
  double np = 0;
  for (int i = 0; i < p.size(); i++)
    np += p[i];
  return np;
}

double KL_divergence(
  std::vector<double> &p, std::vector<double> &q,
  double &h
) {
  // number of elements in p and q ----
  double tp = p.size(), tq = q.size(), kl = 0;
  // Getting bin bounderies ----
  std::vector<double> b = histogram_grid(p, q, h);
  // p and q histograms ----
  std::vector<double> hp = histogram(p, b);
  std::vector<double> hq = histogram(q, b);
  for (int i = 0; i < hp.size(); i++) {
    hp[i] /= tp;
  }
  for (int i = 0; i < hp.size(); i++) {
    hq[i] /= tq;
  }
  // Compute KL ----
  for (int i = 0; i < hp.size(); i++) {
    if (hp[i] == 0) continue;
    kl += hp[i] * log(hp[i] / hq[i]);
  }
  std::cout << "\n";
  return kl;
}

PYBIND11_MODULE(ctools, m) {

  m.doc() = "Creates fast random networks";

  m.def(
    "KL_divergence",
    &KL_divergence,
    py::return_value_policy::reference_internal
  );
}