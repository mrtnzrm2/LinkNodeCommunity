#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include<ctime> // time
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// #define lmbda 0.07921125

std::vector<std::vector<int> > create_id_matrix(
	std::vector<std::vector<bool> >& matrix,
	const int& M, const int& N
) {
	std::vector<std::vector<int> > id_matrix(N, std::vector<int>(M, 0));
	int id = 1;
	for (int i=0; i < M; i++){
		for (int j=0; j < N; j++){
			if (matrix[i][j] > 0){
				id_matrix[i][j] = id;
				id++;
			}
		}
	}
	return id_matrix;
}

class simquest {
  private:
    std::vector<std::vector<double> > linksim_matrix;
    std::vector<std::vector<double> > source_matrix;
    std::vector<std::vector<double> > target_matrix;
  public:
    simquest(
      std::vector<std::vector<bool> > BA,
			std::vector<std::vector<double> > D,
      std::vector<std::vector<double> > AKI,
      std::vector<std::vector<double> > AIK,
			std::vector<std::vector<bool> > BAKI,
  		std::vector<std::vector<bool> > BAIK,
      const int N,
      const int leaves,
      const int topology,
      const int index
    );
    ~simquest(){};
    std::vector<std::vector<double> > calculate_linksim_matrix(
      std::vector<std::vector<bool> >& bmatrix, const int& N, const int& leaves
    );
    double similarity_index(
      std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D, const int &index
    );
    std::vector<std::vector<double> > calculate_nodesim_matrix(
      std::vector<std::vector<double> >& matrix, std::vector<std::vector<bool> >& bmatrix, std::vector<std::vector<double> > D, const int& N, const int& index
    );
		std::vector<std::vector<double> > get_linksim_matrix();
		std::vector<std::vector<double> > get_source_matrix();
		std::vector<std::vector<double> > get_target_matrix();
		double jacp(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv);
		double tanimoto_coefficient(std::vector<double> &u, std::vector<double> &v);
		double cosine_similarity(std::vector<double> &u, std::vector<double> &v);
		double jacw(std::vector<double> &u, std::vector<double> &v);
		double simple(std::vector<double> &u, std::vector<double> &v);
		double simple2(std::vector<double> &u, std::vector<double> &v);
		double from_reg(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D);
		double from_clf(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D);
		double bin_similarity(std::vector<bool> &bu, std::vector<bool> &bv);
};

simquest::simquest(
	std::vector<std::vector<bool> > BA,
	std::vector<std::vector<double>> D,
  std::vector<std::vector<double> > AKI,
  std::vector<std::vector<double> > AIK,
	std::vector<std::vector<bool> > BAKI,
  std::vector<std::vector<bool> > BAIK,
	const int N,
	const int leaves,
  const int topology,
  const int index
){
	// MIX topology
	if (topology == 0) {
		source_matrix = calculate_nodesim_matrix(AIK, BAIK, D, N, index);
		target_matrix = calculate_nodesim_matrix(AKI, BAKI, D, N, index);
	}
	// SOURCE topology
	else if (topology == 1) {
		source_matrix = calculate_nodesim_matrix(AIK, BAIK, D, N, index);
		target_matrix = source_matrix;
	}
	// TARGET topology
	else if (topology == 2) {
		source_matrix = calculate_nodesim_matrix(AKI, BAKI, D, N, index);
		target_matrix = source_matrix;
	}
	linksim_matrix = calculate_linksim_matrix(BA, N, leaves);
}

std::vector<std::vector<double> > simquest::calculate_nodesim_matrix(
	 std::vector<std::vector<double> >& matrix, std::vector<std::vector<bool> >& bmatrix, std::vector<std::vector<double> > D, const int& N, const int& index
) {
	std::vector<std::vector<double> > node_sim_matrix(N, std::vector<double>(N, 0.));
	for (int i=0; i < N; i++) {
		for (int j=i; j < N; j++) {
			if (i == j) continue;
			node_sim_matrix[i][j] = similarity_index(matrix[i], matrix[j], bmatrix[i], bmatrix[j], D[i][j], index);
			node_sim_matrix[j][i] = node_sim_matrix[i][j];
		}
	}
	return node_sim_matrix;
}

std::vector<std::vector<double> > simquest::calculate_linksim_matrix(
	std::vector<std::vector<bool> >& matrix, const int& N, const int& leaves
) {
	std::vector<std::vector<int> > id_matrix = create_id_matrix(matrix, N, N);
	std::vector<std::vector<double> > link_similarity_matrix(leaves, std::vector<double>(leaves, 0));
	int col_id, row_id;
	for (int i =0; i < N; i++) {
		for (int j=0; j < N; j++) {
			row_id = id_matrix[i][j];
			if (row_id == 0) continue;
			for (int k=j; k < N; k++) {
				col_id = id_matrix[i][k];
				if (k == j || col_id == 0) continue;
				link_similarity_matrix[row_id-1][col_id-1] = target_matrix[j][k];
				link_similarity_matrix[col_id-1][row_id-1] = link_similarity_matrix[row_id][col_id];
			}
			for (int k=i; k < N; k++) {
				col_id = id_matrix[k][j];
				if (k == i || col_id == 0) continue;
				link_similarity_matrix[row_id-1][col_id-1] = source_matrix[i][k];
				link_similarity_matrix[col_id-1][row_id-1] = link_similarity_matrix[row_id][col_id];
			}
		}
	}
	return link_similarity_matrix;
}

double simquest::tanimoto_coefficient(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		uv += u[i] * v[i];
		uu += u[i] * u[i];
		vv += v[i] * v[i];
	}
	return uv / (uu + vv - uv);
}

double simquest::cosine_similarity(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		uv += u[i] * v[i];
		uu += u[i] * u[i];
		vv += v[i] * v[i];
	}
	return uv / (sqrt(uu * vv));
}

double simquest::jacw(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double maximus=0.;
	for (int i=0; i < N; i++) {
		maximus += std::min(u[i], v[i]) - std::max(u[i], v[i]);
	}
	return maximus / N;
}

double simquest::simple(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double maximus=0.;
	for (int i=0; i < N; i++)
		maximus -= abs(u[i] - v[i]); 
	return maximus / N;
}

double simquest::simple2(
	std::vector<double> &u, std::vector<double> &v
) {
	int N = u.size();
	double JACP = 0.;
	double p;
	for (int i=0; i < N; i++){
		p = 0;
		for (int j=0; j < N; j++){
			p += std::log(1 + std::max(std::exp(u[j]-u[i]), std::exp(v[j]-v[i])));
		}
		if (p != 0) JACP += std::log(2.) / p;
		else std::cout << "Vectors in jaccardp  are both zero\n";
	}
	if (JACP > 1) {
		JACP = 1 - 1e-8;
		std::cout << "\n\tSimilarity greater than one.\n";
	}
	return JACP;
}

double simquest::bin_similarity(
	std::vector<bool> &bu, std::vector<bool> &bv
) {
	int N = bu.size();
	double uv=0., uu=0., vv=0.;
	for (int i=0; i < N; i++) {
		if ((bu[i] && bv[i]) || (!bu[i] && !bv[i])) uv++;
		if (bu[i]) uu++;
		if (bv[i]) vv++;
	}
	uv /= N;
	uu /= N;
	vv /= N;
	return uv - 1 + uu + vv - (2 * uu * vv);
}

double simquest::jacp(
	std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv)
{
	int N = u.size();
	double JACP = 0;
	double p;
	for (int i=0; i < N; i++){
		if (bu[i] && bv[i]){
			if (u[i] == 0 || v[i] == 0) std::cout << "\n\tWeight-topology ambiguity: Connection exists with zero weigth\n";
			p = 0;
			for (int j=0; j < N; j++){
				p += std::max(u[j]/u[i], v[j]/v[i]);
			}
			if (p != 0)
				JACP += 1/p;
			else
				std::cout << "Vectors in jaccardp are both zero\n";
		}
	}
	return JACP;
}

double simquest::from_reg(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D) {
	double simr = jacp(u, v, bu, bv);
	return -2.36160354e-04 * D + 1.15750346e+01 * simr - 8.46048913e-02 * (simr * D)  - 2.66552556e+00;
}

double simquest::from_clf(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D) {
	double simr = jacp(u, v, bu, bv);
	return 1 / (1 + exp(0.02036713 * D - 8.61685212 * simr - 0.02094556 * (simr * D) + 2.27301252));
}

double simquest::similarity_index(std::vector<double> &u, std::vector<double> &v, std::vector<bool> &bu, std::vector<bool> &bv, double &D, const int &index) {
	// Jaccard probability index
  if (index == 0) {
    return jacp(u, v, bu, bv);
  }
	// Tanimoto coefficient
  else if (index == 1) {
		return tanimoto_coefficient(u, v);
	}
	// Cosine similarity
  else if (index == 2) {
		return cosine_similarity(u, v);
	}
	// Modified weighted Jaccard
	else if (index == 3) {
		return jacw(u, v);
	}
	// binary similarity
	else if (index == 4) {
		return bin_similarity(bu, bv);
	}
	// simple similarity
	else if (index == 5) {
		return simple(u, v);
	}
	else if (index == 6) {
		return simple2(u, v);
	}
	// not simple
	else if (index == 7) {
		return from_reg(u, v, bu, bv, D);
	}
	else if (index == 8) {
		return from_clf(u, v, bu, bv, D);
	}
  else {
    std::range_error("Similarity index must be a integer from 0 to 5\n");
  }
}

std::vector<std::vector<double> > simquest::get_linksim_matrix() {
	return linksim_matrix;
}

std::vector<std::vector<double> > simquest::get_source_matrix() {
	return source_matrix;
}

std::vector<std::vector<double> > simquest::get_target_matrix() {
	return target_matrix;
}

PYBIND11_MODULE(simquest, m) {
    py::class_<simquest>(m, "simquest")
        .def(
          py::init<
           std::vector<std::vector<bool> >,
						std::vector<std::vector<double> >,
						std::vector<std::vector<double> >,
						std::vector<std::vector<double> >,
						std::vector<std::vector<bool> >,
						std::vector<std::vector<bool> >,
						const int,
						const int,
						const int,
						const int
          >()
        )
        .def("get_linksim_matrix", &simquest::get_linksim_matrix)
        .def("get_source_matrix", &simquest::get_source_matrix)
				.def("get_target_matrix", &simquest::get_target_matrix)
				.def("jacp", &simquest::jacp)
        .def("jacw", &simquest::jacw)
				.def("simple", &simquest::simple)
				.def("simple2", &simquest::simple2)
				.def("from_reg", &simquest::from_reg)
				.def("from_clf", &simquest::from_clf)
				.def("cosine_similarity", &simquest::cosine_similarity)
			  .def("tanimoto_coefficient", &simquest::tanimoto_coefficient)
        .def("bin_similarity", &simquest::bin_similarity);
}