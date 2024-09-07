#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include<ctime> // time
#include "hclust-cpp/fastcluster.h"

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

double simbin(std::vector<double> &bu, std::vector<double> &bv
) {
	int N = bu.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		if ((bu[i] > 0 && bv[i] > 0) || (bu[i] == 0 && bv[i] == 0)) uv++;
		if (bu[i] > 0) uu++;
		if (bv[i] > 0) vv++;
	}
	uv /= N;
	uu /= N;
	vv /= N;
	return uv - 1 + uu + vv - (2 * uu * vv);
}

double Hellinger2(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size(), k=0;
	double p = 0, pu = 0., pv = 0., maxp, s;
	std::vector<bool> possible;
	std::vector<double> ppu(N, 0.), ppv(N, 0.), peff(N, 0.);

	for (int j=0; j < N; j++){
		pu += u[j];
		pv += v[j];

		if (j == ii || j == jj) continue;
		ppu[k] = u[j];
		ppv[k] = v[j];
		k++;
	}
	ppu[N-2] = u[ii];
	ppu[N-1] = u[jj];
	ppv[N-2] = v[jj];
	ppv[N-1] = v[ii];

	if (pu == 0 || pv == 0) return  0.;

	for (int j=0; j < N; j++) {
		if (ppu[j] > 0 && ppv[j] > 0) {
			peff[j] = 0.5 * (log(ppu[j]) + log(ppv[j]) - log(pu) -log(pv));
			possible.push_back(true);
		}	else {
			possible.push_back(false);
		}
		
	}
	
	for (int j=0; j < N; j++) {
		if (possible[j])  {
			if (maxp > peff[j]) maxp = peff[j];
		}
	}
	if (maxp == 0) return 0.;

	for (int j=0; j < N; j++) {
		if (possible[j]) {
			s += exp(peff[j]-maxp);
		}
	}
	return s * exp(maxp);
}

double cosine_similarity(
	std::vector<double> &u, std::vector<double> &v, int &ii, int &jj
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		uu += u[i] * u[i];
		vv += v[i] * v[i];
		if (i == ii | i == jj) continue;
		uv += u[i] * v[i];
	}

	if (ii < N && jj < N) {
		uv += u[jj] * v[ii];
		uv += u[ii] * v[jj]; 
	}

	return uv / (sqrt(uu * vv));
}

PYBIND11_MODULE(ctools, m) {

  m.doc() = "Creates fast random networks";

  m.def(
    "KL_divergence",
    &KL_divergence,
    py::return_value_policy::reference_internal
  );

  m.def(
    "simbin",
    &simbin,
    py::return_value_policy::reference_internal
  );

	m.def(
    "Hellinger2",
    &Hellinger2,
    py::return_value_policy::reference_internal
  );

	m.def(
    "cosine_similarity",
    &cosine_similarity,
    py::return_value_policy::reference_internal
  );
}