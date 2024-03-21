#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include<ctime> // time

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<double> > >);
// using TensorDouble = std::vector<std::vector<std::vector<double> > >;

// template<typename T>
std::vector<std::vector<long int> > create_id_matrix(
	std::vector<std::vector<bool> >& matrix,
	const int& M, const int& N
) {
	std::vector<std::vector<long int> > id_matrix(N, std::vector<long int>(M, 0));
	long int id = 1;
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

class signsimquest {
	private:
		int arch;
  public:
		py::array_t<double> linksim_matrix;
    std::vector<std::vector<double> > source_matrix;
    std::vector<std::vector<double> > target_matrix;
    signsimquest(
			std::vector<std::vector<bool> > BA,
      std::vector<std::vector<double> > A,
			std::vector<std::vector<double> > D,
      py::array_t<double> &AKI,
      py::array_t<double> &AIK,
      const int N,
      const int leaves,
      const int topology,
      const int index,
			const int architecture
    );
    ~signsimquest(){};
    void calculate_linksim_matrix(
			std::vector<std::vector<double> >& matrix, std::vector<std::vector<bool> >& bmatrix, const int& N, const int& leaves
    );
    double similarity_index(
      py::array_t<double> &u, double &D, int &ii, int &jj, const int &index
    );
    std::vector<std::vector<double> > calculate_nodesim_matrix(
      py::array_t<double>& matrix, std::vector<std::vector<double> > D, const int& N, const int& index
    );
		double Hellinger2(py::array_t<double> &u, int &ii, int&jj);
};

signsimquest::signsimquest(
	std::vector<std::vector<bool> > BA,
	std::vector<std::vector<double> > A,
	std::vector<std::vector<double>> D,
  py::array_t<double>  &AKI,
  py::array_t<double>  &AIK,
	const int N,
	const int leaves,
  const int topology,
  const int index,
	const int architecture
){
	arch = architecture;
	// MIX topology
	if (topology == 0) {
		source_matrix = calculate_nodesim_matrix(AIK, D, N, index);
		target_matrix = calculate_nodesim_matrix(AKI, D, N, index);
	}
	// SOURCE topology
	else if (topology == 1) {
		source_matrix = calculate_nodesim_matrix(AIK, D, N, index);
		target_matrix = source_matrix;
	}
	// TARGET topology
	else if (topology == 2) {
		source_matrix = calculate_nodesim_matrix(AKI, D, N, index);
		target_matrix = source_matrix;
	}
	calculate_linksim_matrix(A, BA, N, leaves);
}

std::vector<std::vector<double> > signsimquest::calculate_nodesim_matrix(
	 py::array_t<double> &matrix, std::vector<std::vector<double> > D, const int& N, const int& index
) {

	std::vector<std::vector<double> > node_sim_matrix(N, std::vector<double>(N, 0.));
	for (int i=0; i < N; i++) {
		for (int j=i; j < N; j++) {
			if (i == j) continue;
			node_sim_matrix[i][j] = similarity_index(matrix, D[i][j], i, j, index);
			node_sim_matrix[j][i] = node_sim_matrix[i][j];
		}
	}
	return node_sim_matrix;
}

void signsimquest::calculate_linksim_matrix(
	std::vector<std::vector<double> >& matrix, std::vector<std::vector<bool> >& bmatrix, const int& N, const int& leaves
) {
	long int t, col_id, row_id;
	std::vector<std::vector<long int> > id_matrix = create_id_matrix(bmatrix, N, N);

	t = (long int) ((leaves - 1.) * leaves / 2.);
	linksim_matrix = py::array_t<double>(t);
  py::buffer_info bufr = linksim_matrix.request();
	double *ptr = (double *) bufr.ptr;

	// py::print(arch);

	for (int i=0; i < t; i++)
		ptr[i] = 0.;
	for (int i =0; i < N; i++) {
		for (int j=0; j < N; j++) {
			row_id = id_matrix[i][j];
			if (row_id == 0) continue;
			for (int k=j; k < N; k++) {
				col_id = id_matrix[i][k];
				if (k == j || col_id == 0) continue;
				if (arch == 0) {
					if (matrix[i][j] * matrix[i][k] <= 0)
						continue;}
				else if (arch == 1){
					if (matrix[i][j] <= 0 || matrix[i][k] <= 0)
						continue;}
				else if (arch == 2){
					if (matrix[i][j] >= 0 || matrix[i][k] >= 0)
						continue;}
				else if (arch == 3){
					if (matrix[i][j] * matrix[i][k] >= 0)
						continue;}
				else if (arch == 4)
					;
				else
					throw std::invalid_argument("No architecture");
				t = leaves * (row_id - 1) + col_id - 1 - 2 * (row_id -1) - 1;
				t -= (long int) ((row_id - 1.) * (row_id - 2.) / 2);
				ptr[t] = target_matrix[j][k];
			}
			for (int k=i; k < N; k++) {
				col_id = id_matrix[k][j];
				if (k == i || col_id == 0) continue;
				if (arch == 0){
					if (matrix[i][j] * matrix[k][j] <= 0)
						continue;}
				else if (arch == 1){
					if (matrix[i][j] <= 0 || matrix[k][j] <= 0)
						continue;}
				else if (arch == 2){
					if (matrix[i][j] >= 0 || matrix[k][j] >= 0)
						continue;}
				else if (arch == 3){
					if (matrix[i][j] * matrix[k][j] >= 0)
						continue;}
				else if (arch == 4)
					;
				else
					throw std::invalid_argument("No architecture");
				t = leaves * (row_id - 1) + col_id - 1 - 2 * (row_id -1) - 1;
				t -= (long int) ((row_id - 1.) * (row_id - 2.) / 2);
				ptr[t] = source_matrix[i][k];
			}
		}
	}
	// for (int i=0; i < bufr.size; i++)
	// 	std::cout << ptr[i] << " ";
	// std::cout << "\n";

}

// double signsimquest::Hellinger2(
// 	py::array_t<double> &F, int &ii, int &jj
// ) {

// 	py::buffer_info buf1 = F.request();
// 	double *ptrF = (double *) buf1.ptr;

// 	int L = buf1.shape[0];
// 	int N = buf1.shape[1];
// 	int M = buf1.shape[2];
	
// 	double p = 0, pu = 0, pv = 0;

// 	for (size_t j=0; j < N; j++){
// 		for (size_t k=0; k < M; k++) {
// 			pu += ptrF[ii*N*M + j*M + k];
// 			pv += ptrF[jj*N*M + j*M + k];
// 		}
// 	}
// 	if (pu > 0 && pv > 0) {
// 		for (size_t i=0; i < N; i++){
// 			if (i == ii | i == jj) continue;
// 			for (size_t j=0; j < M; j++)
// 				p += pow(sqrt(ptrF[ii*N*M + i*M + j] / pu)  - sqrt(ptrF[jj*N*M + i*M + j] / pv), 2.);
// 		}
// 		if (ii < N && jj < N) {
// 			for (size_t j=0; j < M; j++) {
// 				p += pow(sqrt(ptrF[ii*N*M + jj*M + j] / pu) - sqrt(ptrF[jj*N*M + ii*M + j] / pv), 2.);
// 				p += pow(sqrt(ptrF[ii*N*M + ii*M + j] / pu) - sqrt(ptrF[jj*N*M + jj*M + j] / pv), 2.);
// 			}
// 		}
// 		return 1. - (0.5 * p);
// 	}
// 	else {
// 		return 0.;
// 	}
// }

double signsimquest::Hellinger2(
	py::array_t<double> &F, int &ii, int &jj
) {

	py::buffer_info buf1 = F.request();
	std::vector<bool> possible;
	double *ptrF = (double *) buf1.ptr;

	int L = buf1.shape[0];
	int N = buf1.shape[1];
	int M = buf1.shape[2];
	
	double p = 0, pu = 0, pv = 0, w = 0, s, maxp;

	std::vector<double> ppu(N*M,  0.), ppv(N*M, 0.), peff(N*M, 0.);

	for (size_t j=0; j < N; j++){
		for (size_t k=0; k < M; k++) {
			pu += ptrF[ii*N*M + j*M + k];
			pv += ptrF[jj*N*M + j*M + k];

			if (j == ii || j == jj) continue;
			ppu[w] = ptrF[ii*N*M + j*M + k];
			ppv[w] = ptrF[jj*N*M + j*M + k];
			w++;
		}
	}

	for (size_t k=0; k < M; k++) {
		ppu[w] = ptrF[ii*N*M + jj*M + k];
		ppv[w] = ptrF[jj*N*M + ii*M + k];
		w++;
	}

	for (size_t k=0; k < M; k++) {
		ppu[w] = ptrF[ii*N*M + ii*M + k];
		ppv[w] = ptrF[jj*N*M + jj*M + k];
		w++;
	}
	
	if (pu == 0 || pv == 0) return  0.;

	for (int j=0; j < N*M; j++) {
		if (ppu[j] > 0 && ppv[j] > 0) {
			peff[j] = 0.5 * (log(ppu[j]) + log(ppv[j]) - log(pu) -log(pv));
			possible.push_back(true);
		}	else{
			possible.push_back(false);
		}
		
	}
	
	for (int j=0; j < N*M; j++) {
		if (possible[j])  {
			if (maxp > peff[j]) maxp = peff[j];
		}
	}
	if (maxp == 0) return 0.;

	for (int j=0; j < N*M; j++) {
		if (possible[j]) {
			s += exp(peff[j]-maxp);
		}
	}
	return s * exp(maxp);
}

double signsimquest::similarity_index(py::array_t<double> &u, double &D, int &ii, int &jj, const int &index) {
	// Jaccard probability index
  if (index == 0) {
		return Hellinger2(u, ii, jj);
	}
  else {
    std::range_error("Similarity index must be a integer from 0 to 5\n");
  }
}

PYBIND11_MODULE(signsimquest, m) {

    py::class_<signsimquest>(m, "signsimquest")
        .def(
          py::init<
						std::vector<std::vector<bool> >,
           	std::vector<std::vector<double> >,
						std::vector<std::vector<double> >,
						py::array_t<double> &,
						py::array_t<double> &,
						const int,
						const int,
						const int,
						const int,
						const int
          >()
        )
        // .def("get_linksim_matrix", &signsimquest::get_linksim_matrix)
        // .def("get_source_matrix", &signsimquest::get_source_matrix)
				// .def("get_target_matrix", &signsimquest::get_target_matrix);
				.def_readwrite("linksim_matrix", &signsimquest::linksim_matrix)
				.def_readwrite("source_matrix", &signsimquest::source_matrix)
				.def_readwrite("target_matrix", &signsimquest::target_matrix);
}