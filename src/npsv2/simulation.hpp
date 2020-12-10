#pragma once
#include <pybind11/pybind11.h>

#include "SeqLib/UnalignedSequence.h"

namespace py = pybind11;
namespace sl = SeqLib;

namespace npsv2 {
void FilterReadsGC(const std::string& fasta_path, const std::string& sam_path, const std::string& fastq_path,
                   const std::vector<float>& gc_covg);
}
