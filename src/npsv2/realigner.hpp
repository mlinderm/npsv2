#pragma once

#include <pybind11/pybind11.h>

#include "SeqLib/BWAWrapper.h"
#include "SeqLib/BamReader.h"

namespace py = pybind11;
namespace sl = SeqLib;

namespace npsv2 {

class IndexedSequence;

class InsertSizeDistribution {
 public:
  typedef std::map<int, double> density_type;

  InsertSizeDistribution() {}
  InsertSizeDistribution(double mean, double std) : mean_(mean), std_(std) {}
  InsertSizeDistribution(double mean, double std, const density_type& density)
      : mean_(mean), std_(std), density_(density) {}

  double operator()(int insert_size) const;

  double ZScore(int insert_size) const { return (insert_size - mean_) / std_; }

 private:
  double mean_, std_;
  density_type density_;
};

class RealignedReadPair {
 public:
  typedef double score_type;

  RealignedReadPair() : left_(nullptr), right_(nullptr), score_(0) {}
  RealignedReadPair(const sl::BamRecord& first);
  RealignedReadPair(const sl::BamRecord& first, const sl::BamRecord& second_, const InsertSizeDistribution&);

  bool IsValid() const { return left_ || right_; }
  score_type Score() const { return score_; }

  bool operator<(const RealignedReadPair& other) const { return score_ < other.score_; }
  bool operator>(const RealignedReadPair& other) const { return score_ > other.score_; }

  sl::GenomicRegion FragmentRegion() const;

  friend std::ostream& operator<<(std::ostream&, const RealignedReadPair&);

 private:
  const sl::BamRecord* left_;
  const sl::BamRecord* right_;
  score_type score_;

  bool Concordant() const;
  int32_t InsertSize() const;
};

class RealignedFragment {
  typedef std::vector<RealignedReadPair> PairSequence;

 public:
  typedef RealignedReadPair::score_type score_type;

  RealignedFragment(const sl::BamRecord& read1, const sl::BamRecord& read2, const IndexedSequence&,
                    const InsertSizeDistribution&, int quality_offset = 33);

  PairSequence::size_type NumAlignments() const { return read_pairs_.size(); }

  bool HasBestPair() const { return !read_pairs_.empty(); }
  const RealignedReadPair& BestPair() const { return read_pairs_.front(); }
  score_type BestPairLogProb() const { return HasBestPair() ? BestPair().Score() : 0.; }

  score_type TotalLogProb() const { return total_log_prob_; }

 private:
  sl::BamRecordVector read1_alignments_;
  sl::BamRecordVector read2_alignments_;
  PairSequence read_pairs_;
  score_type total_log_prob_;
};

class IndexedSequence {
 public:
  IndexedSequence() {}
  IndexedSequence(const sl::UnalignedSequence& sequence);

  bool IsInitialized() const { return !bwa_.IsEmpty(); }
  void Initialize(const sl::UnalignedSequence& sequence);

  const std::string& Sequence() const { return sequence_.Seq; }
  sl::BamHeader Header() const { return bwa_.HeaderFromIndex(); }

  void AlignSequence(const sl::BamRecord& read, sl::BamRecordVector& alignments) const;
  void AlignSequence(const std::string& name, const std::string& seq, sl::BamRecordVector& alignments) const;

 private:
  sl::UnalignedSequence sequence_;
  sl::BWAWrapper bwa_;
};

class FragmentRealigner {
  typedef std::vector<IndexedSequence> AltIndexesSequence;

 public:
  FragmentRealigner(const std::string& fasta_path, double insert_size_mean, double insert_size_std);

  AltIndexesSequence::size_type NumAltAlleles() const { return alt_indexes_.size(); }

  std::tuple<double, bool, double, bool> RealignReadPair(const std::string& name, const std::string& read1_seq,
                                                const std::string& read1_qual, py::kwargs kwargs);

 private:
  InsertSizeDistribution insert_size_dist_;
  IndexedSequence ref_index_;
  std::vector<IndexedSequence> alt_indexes_;

  sl::BamHeader RefHeader() const { return ref_index_.Header(); }
  sl::BamHeader AltHeader(int index) const { return alt_indexes_[index].Header(); }
};

namespace test {

std::vector<double> TestScoreAlignment(const std::string& ref_seq, const std::string& aln_path);

std::tuple<double, bool, double, bool> TestRealignReadPair(const std::string& fasta_path, const std::string& name,
                                                  const std::string& read1_seq, const std::string& read1_qual,
                                                  py::kwargs kwargs);
}  // namespace test

}  // namespace npsv2