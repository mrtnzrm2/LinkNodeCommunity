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

double jacp(std::vector<double> &u, std::vector<double> &v) {
	int N = u.size();
	double JACP = 0;
	double p;
	for (int i=0; i < N; i++){
		if (u[i] > 0 && v[i] > 0){
			p = 0;
			for (int j=0; j < N; j++) {
				p += std::max(u[j]/u[i], v[j]/v[i]);
			}
      JACP += 1/p;
		}
	}
	return JACP;
}

double jaclog(std::vector<double> &u, std::vector<double> &v) {
	int N = u.size();
	double JACP = 0.;
	double p;
	for (int i=0; i < N; i++){
		p = 0;
		for (int j=0; j < N; j++){
			p += std::log(1 + std::max((1 + u[j]) / (1 + u[i]), (1 + v[j]) / (1 + v[i])));
		}
		if (p != 0) JACP += std::log(2.) / p;
		else std::cout << "Vectors in jaccardp  are both zero\n";
	}
	return JACP;
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

double jacsqrt(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p;
	for (int i=0; i < N; i++){
		p = 0;
		for (int j=0; j < N; j++){
			p += std::sqrt(std::max((1 + u[j]) / (1 + u[i]), (1 + v[j]) / (1 + v[i])));
		}
		if (p != 0) JACP += 1 / p;
		else std::cout << "Vectors in jaccardp  are both zero\n";
	}
	return JACP;
}

double D1(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
			pu += 1 + u[j];
			pv += 1 + v[j];
	}
	for (int i=0; i < N; i++){
		// D1
		p += ((1 + u[i]) / pu) * log(((1 + u[i]) / pu) * (pv / (1 + v[i])));
		p += ((1 + v[i]) / pv) * log(((1 + v[i]) / pv) * (pu / (1 + u[i])));
	}
	JACP = 1 /(1 + p / 2.);
	return JACP;
}

double D1b(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
			pu += u[j];
			pv += v[j];
	}
	for (int i=0; i < N; i++){
		// D1
		// if (u[i] == 0 || v[i] == 0) continue;
		p += ((u[i]) / pu) * log(((u[i]) / pu) * (pv / (v[i])));
		p += ((v[i]) / pv) * log(((v[i]) / pv) * (pu / (u[i])));
	}
	if (p >= 0)
		JACP = 1 /(1 + p / 2.);
	else
		JACP = 0.;
	return JACP;
}

double D1_2(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
			pu += 1 + u[j];
			pv += 1 + v[j];
	}
	for (int i=0; i < N; i++){
		// D1/2
		p += sqrt(((1 + u[i]) / pu) * ((1 + v[i]) / pv));
	}
	// D1/2
	 p = - 2 * log(p);
	JACP = 1 / (1 + p);
	return JACP;
}

double D2(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0, p2 = 0;
	for (int j=0; j < N; j++){
			pu += 1 + u[j];
			pv += 1 + v[j];
	}
	for (int i=0; i < N; i++){
		// D2
		p += pow((1 + u[i]) / pu, 2.) * (pv / (1 + v[i]));
		p2 += pow((1 + v[i]) / pv, 2.) * (pu / (1 + u[i]));
	}
	// D2
	p = log(p) + log(p2);
	JACP = 1 /(1 + p / 2.);
	return JACP;
}

double Dinf(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0, p2 = 0;
	for (int j=0; j < N; j++){
			pu += 1 + u[j];
			pv += 1 + v[j];
	}
	for (int i=0; i < N; i++){
		// Dinf
		if (((1 + u[i]) / pu) * (pv / (1 + v[i])) > p) p = ((1 + u[i]) / pu) * (pv / (1 + v[i]));
		if (((1 + v[i]) / pv) * (pu / (1 + u[i])) > p2) p2 = ((1 + v[i]) / pv) * (pu / (1 + u[i]));
	}
	// Dinf
	p = log(p) + log(p2);
	JACP = 1 /(1 + p / 2.);
	return JACP;
}

double D1_2_2(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0;
	for (int j=0; j < N; j++){
			pu += u[j];
			pv += v[j];
	}
	for (int i=0; i < N; i++){
		// D1/2
		p += sqrt(((u[i]) / pu) * ((v[i]) / pv));
	}
	// D1/2
  if (p > 0) {
    p = - 2 * log(p);
    JACP = 1 / (1 + p);
  }
  else
    JACP = 0;
	return JACP;
}

double Dalpha(
	std::vector<double> &u, std::vector<double> &v, double &alpha
) {
	int N = u.size();
	double JACP = 0.;
	double p = 0, pu = 0, pv = 0, q = 0;
	for (int j=0; j < N; j++){
			pu += 1 + u[j];
			pv += 1 + v[j];
	}
	for (int i=0; i < N; i++){
		p += pow((1 + u[i]) / pu, alpha) / pow((1 + v[i]) / pv, alpha - 1);
		q += pow((1 + v[i]) / pv, alpha) / pow((1 + u[i]) / pu, alpha - 1);
	}
	// Dalpha
	JACP = log(p) / (alpha - 1) + log(q) / (alpha - 1);
	JACP = 1 /(1 + JACP / 2.);
	return JACP;
}

PYBIND11_MODULE(ctools, m) {

  m.doc() = "Creates fast random networks";

  m.def(
    "KL_divergence",
    &KL_divergence,
    py::return_value_policy::reference_internal
  );

  m.def(
    "jaclog",
    &jaclog,
    py::return_value_policy::reference_internal
  );

  m.def(
    "jacp",
    &jacp,
    py::return_value_policy::reference_internal
  );

  m.def(
    "simbin",
    &simbin,
    py::return_value_policy::reference_internal
  );

   m.def(
    "jacsqrt",
    &jacsqrt,
    py::return_value_policy::reference_internal
  );

  m.def(
    "D1",
    &D1,
    py::return_value_policy::reference_internal
  );

  m.def(
    "D1_2",
    &D1_2,
    py::return_value_policy::reference_internal
  );

  m.def(
    "D1b",
    &D1b,
    py::return_value_policy::reference_internal
  );

  m.def(
    "D2",
    &D2,
    py::return_value_policy::reference_internal
  );

  m.def(
    "Dinf",
    &Dinf,
    py::return_value_policy::reference_internal
  );

  m.def(
    "Dalpha",
    &Dalpha,
    py::return_value_policy::reference_internal
  );

  m.def(
    "D1_2_2",
    &D1_2_2,
    py::return_value_policy::reference_internal
  );
}