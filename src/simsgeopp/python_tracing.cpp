#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "py_shared_ptr.h"
PYBIND11_DECLARE_HOLDER_TYPE(T, py_shared_ptr<T>);
using std::shared_ptr;
using std::vector;
#include "tracing.h"


void init_tracing(py::module_ &m){


    py::class_<StoppingCriterion, shared_ptr<StoppingCriterion>>(m, "StoppingCriterion");
    py::class_<IterationStoppingCriterion, shared_ptr<IterationStoppingCriterion>, StoppingCriterion>(m, "IterationStoppingCriterion")
        .def(py::init<int>());
    py::class_<LevelsetStoppingCriterion<PyArray>, shared_ptr<LevelsetStoppingCriterion<PyArray>>, StoppingCriterion>(m, "LevelsetStoppingCriterion")
        .def(py::init<shared_ptr<RegularGridInterpolant3D<PyArray>>>());

    m.def("particle_guiding_center_tracing", py::overload_cast<shared_ptr<MagneticField<PyArray>>, double, double, double,
        double, double, double, double, double, double, vector<shared_ptr<StoppingCriterion>>>(&particle_guiding_center_tracing<PyArray>));
    m.def("particle_guiding_center_tracing", py::overload_cast<shared_ptr<MagneticField<PyArray>>, double, double, double,
        double, double, double, double, double, double>(&particle_guiding_center_tracing<PyArray>));
    m.def("fieldline_tracing", &fieldline_tracing<PyArray>, 
            py::arg("field"),
            py::arg("xinit"),
            py::arg("yinit"),
            py::arg("zinit"),
            py::arg("tmax"),
            py::arg("tol"),
            py::arg("phis"),
            py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{});
}
