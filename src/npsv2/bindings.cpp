#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "realigner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_realigner, m) {
  m.doc() = "NPSV2 re-alignment tools";

  py::class_<npsv2::FragmentRealigner>(m, "FragmentRealigner")
      .def(py::init<const std::string&, double, double>())
      .def("realign_read_pair", &npsv2::FragmentRealigner::RealignReadPair);

  m.def("test_score_alignment", &npsv2::test::TestScoreAlignment, "Test interface for scoring alignment");
  m.def("test_realign_read_pair", &npsv2::test::TestRealignReadPair, "Test interface for realigning reads");
}