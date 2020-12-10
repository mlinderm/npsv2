#include "simulation.hpp"
#include "utility.hpp"

#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "SeqLib/BamReader.h"
#include "SeqLib/BamWriter.h"
#include "SeqLib/FastqReader.h"
#include "SeqLib/SeqLibUtils.h"
#include "SeqLib/UnalignedSequence.h"
#include "utility.hpp"

namespace npsv2 {

void WriteFastQRead(std::ofstream& writer, const sl::BamRecord& read, int read_num) {
  writer << "@" << read.Qname() << "/" << read_num << std::endl;
  if (read.ReverseFlag()) {
    std::string sequence(read.Sequence()), qualities(read.Qualities());
    sl::rcomplement(sequence);
    std::reverse(qualities.begin(), qualities.end());
    writer << sequence << std::endl << "+" << std::endl << qualities << std::endl;
  } else {
    writer << read.Sequence() << std::endl << "+" << std::endl << read.Qualities() << std::endl;
  }
}

void WriteFastQ(std::ofstream& writer, const sl::BamRecord& read1, const sl::BamRecord& read2) {
  if (read1.FirstFlag()) {
    WriteFastQRead(writer, read1, 1);
    WriteFastQRead(writer, read2, 2);
  } else {
    WriteFastQRead(writer, read2, 1);
    WriteFastQRead(writer, read1, 2);
  }
}

constexpr bool IsGC(char base) { return base == 'G' || base == 'C' || base == 'g' || base == 'c'; }

void FilterReadsGC(const std::string& fasta_path, const std::string& sam_path, const std::string& fastq_path,
                   const std::vector<float>& gc_covg) {
  pyassert(gc_covg.size() == 101, "GC vector should have entries for 0-100");

  // Open the input SAM file and any output files
  sl::BamReader reader;
  reader.Open(sam_path);

  std::ofstream writer(fastq_path);

  // Load alleles from a FASTA file
  std::vector<std::vector<int> > contigs;
  {
    const auto& header = reader.Header();
    contigs.resize(header.NumSequences());

    sl::FastqReader contig_reader(fasta_path);
    sl::UnalignedSequence next_sequence;
    while (contig_reader.GetNextSequence(next_sequence)) {
      auto& sequence = next_sequence.Seq;
      auto& cuml_gc_count = contigs[header.Name2ID(next_sequence.Name)];

      // Precompute the GC count in each sequence forming vector of cumulative
      // gc counts that can be used to calculate GC fraction for fragments
      cuml_gc_count.resize(sequence.size() + 1);
      cuml_gc_count[0] = 0;
      for (int i = 0; i < sequence.size(); i++) {
        if (IsGC(sequence[i]))
          cuml_gc_count[i + 1] = cuml_gc_count[i] + 1;
        else
          cuml_gc_count[i + 1] = cuml_gc_count[i];
      }
    }
  }

  // Setup random number generator
  std::default_random_engine engine;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  sl::BamRecord read1, read2;
  while (true) {
    if (!reader.GetNextRecord(read1)) break;
    pyassert(reader.GetNextRecord(read2), "Missing second read in pair");

    // Compute GC fraction for insert
    auto& cuml_gc_count = contigs[read1.ChrID()];

    // Due to insertions it is possible for the fragment to extend beyond the legnth of the sequence, so
    // clamp at the sequence length
    auto start = std::min(read1.Position(), read2.Position());
    pyassert(start < cuml_gc_count.size(), "Fragment start coordinate outside of sequence");

    auto length = std::abs(read1.InsertSize());
    if (start + length >= cuml_gc_count.size()) length = cuml_gc_count.size() - start - 1;

    int gc = cuml_gc_count[start + length] - cuml_gc_count[start];
    int gc_fraction = std::lround(static_cast<float>(gc * 100) / length);
    pyassert(gc_fraction >= 0 && gc_fraction <= 100, "GC fraction outside of expected range");

    // Downsample reads based on GC normalized coverage
    float gc_norm_covg = gc_covg[gc_fraction];
    if (dist(engine) < gc_norm_covg) WriteFastQ(writer, read1, read2);
  }
}

}  // namespace npsv2