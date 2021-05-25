#pragma once

#include <pybind11/pybind11.h>

#include "SeqLib/BWAWrapper.h"
#include "SeqLib/BamReader.h"

namespace py = pybind11;
namespace sl = SeqLib;

namespace npsv2 {

class IndexedSequence;

enum GenomicRegionOverlap {
  NoOverlap = 0,
  PartialOverlap = 1,
  ContainsArg = 2,
  ContainedInArg = 3
};

class InsertSizeDistribution {
 public:
  typedef std::map<int, double> density_type;

  InsertSizeDistribution() {}
  InsertSizeDistribution(double mean, double std) : mean_(mean), std_(std) {}
  InsertSizeDistribution(double mean, double std, const density_type& density)
      : mean_(mean), std_(std), density_(density) {}

  double operator()(int insert_size) const;

  double ZScore(int insert_size) const { return (insert_size - mean_) / std_; }
  double Max() const;

 private:
  double mean_, std_;
  density_type density_;
};

class RealignedReadPair {
 public:
  typedef double score_type;

  RealignedReadPair() : left_(nullptr), right_(nullptr), score_(0.), max_score_(0.) {}
  RealignedReadPair(const sl::BamRecord& first);
  RealignedReadPair(const sl::BamRecord& first, const sl::BamRecord& second_, const InsertSizeDistribution&);

  bool IsValid() const { return left_ || right_; }
  score_type Score() const { return score_; }
  score_type MaxPossibleScore() const { return max_score_; }

  bool operator<(const RealignedReadPair& other) const { return score_ < other.score_; }
  bool operator>(const RealignedReadPair& other) const { return score_ > other.score_; }

  sl::GenomicRegion FragmentRegion() const;

  friend std::ostream& operator<<(std::ostream&, const RealignedReadPair&);

 private:
  const sl::BamRecord* left_;
  const sl::BamRecord* right_;
  score_type score_;
  score_type max_score_;

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

  const std::string& IUPACSequence() const;
  void SetIUPACSequence(const sl::UnalignedSequence& sequence) { iupac_sequence_ = sequence; }

  void AlignSequence(const sl::BamRecord& read, sl::BamRecordVector& alignments) const;
  void AlignSequence(const std::string& name, const std::string& seq, sl::BamRecordVector& alignments) const;

 private:
  sl::UnalignedSequence sequence_;
  sl::BWAWrapper bwa_;
  sl::UnalignedSequence iupac_sequence_;
};

class FragmentRealigner {
  typedef std::vector<IndexedSequence> AltIndexesSequence;
 public:  
  typedef std::vector<std::tuple<std::string, std::string, std::string, std::string>> BreakpointList;
  typedef std::tuple<double, bool, double, double, double, bool, double, double> RealignTuple;

  FragmentRealigner(const std::string& fasta_path, const BreakpointList& breakpoints, double insert_size_mean, double insert_size_std, py::kwargs kwargs);

  AltIndexesSequence::size_type NumAltAlleles() const { return alt_indexes_.size(); }

  RealignTuple RealignReadPair(const std::string& name, const std::string& read1_seq,
                                                const std::string& read1_qual, py::kwargs kwargs);

 private:
  std::vector<std::array<sl::GenomicRegion, 4> > breakpoints_;
  InsertSizeDistribution insert_size_dist_;
  IndexedSequence ref_index_;
  AltIndexesSequence alt_indexes_;

  sl::BamHeader RefHeader() const { return ref_index_.Header(); }
  sl::BamHeader AltHeader(int index) const { return alt_indexes_[index].Header(); }
};

namespace test {

std::vector<double> TestScoreAlignment(const std::string& ref_seq, const std::string& aln_path);

FragmentRealigner::RealignTuple TestRealignReadPair(const std::string& fasta_path, const FragmentRealigner::BreakpointList& breakpoints, const std::string& name,
                                                  const std::string& read1_seq, const std::string& read1_qual,
                                                  py::kwargs kwargs);
}  // namespace test

}  // namespace npsv2