#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "realigner.hpp"
#include "simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "NPSV2 native tools";

  py::class_<npsv2::FragmentRealigner>(m, "FragmentRealigner")
      .def(py::init<const std::string&, const npsv2::FragmentRealigner::BreakpointList&, double, double, py::kwargs>())
      .def("realign_read_pair", &npsv2::FragmentRealigner::RealignReadPair);

  m.def("filter_reads_gc", &npsv2::FilterReadsGC, "Filter reads based on GC normalized coverage");
  m.def("filter_reads_gnomad", &npsv2::FilterReadsGnomAD, "Filter reads based on GnomAD normalized coverage");

  m.def("test_score_alignment", &npsv2::test::TestScoreAlignment, "Test interface for scoring alignment");
  m.def("test_realign_read_pair", &npsv2::test::TestRealignReadPair, "Test interface for realigning reads");
}