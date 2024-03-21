#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include<ctime> // time

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<double> > >);
using TensorDouble = std::vector<std::vector<std::vector<double> > >;


class MyClass {
	public:
    TensorDouble contents;
		TensorDouble contents2;
		std::vector<std::vector<double> > sum_c();
};

std::vector<std::vector<double> > MyClass::sum_c() {
	std::vector<std::vector<double> > v(contents.size(), std::vector<double>(contents[0].size(), 0.));

	for(int i=0; i < contents.size(); i++) {
		for (int j=0; j < contents[0].size(); j++) {
			for (int k=0; k < contents[0][0].size(); k++) {
				v[i][j] += contents[i][j][k] + contents2[i][j][k];
			}
		}
	}

	return v;
}

PYBIND11_MODULE(toybind, m)
 {

  m.doc() = "Only to practice";

	py::bind_vector<TensorDouble>(m, "tensorDouble");

	py::class_<MyClass>(m, "MyClass")
    .def(py::init<>())
		.def("sum_c", &MyClass::sum_c)
    .def_readwrite("contents", &MyClass::contents)
		.def_readwrite("contents2", &MyClass::contents2);


}