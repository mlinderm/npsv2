#include "realigner.hpp"

#include "SeqLib/FastqReader.h"
#include "utility.hpp"

namespace {

double LogSumPow(double acc, double prob) {
  double diff = prob - acc;
  if (diff > 100)
    return prob;
  else if (diff < -100)
    return acc;
  else
    return acc + log10(1 + pow(10., diff));
}

double PhredToProb(double phred) { return pow(10.0, phred / -10.0); }

double PhredToLogProb(double quality, double penalty = 0.) { return (-quality / 10.) + penalty; }

double LogProbToPhredQual(double prob, double max_qual) {
  return std::min(log10(1. - pow(10.0, prob)) * -10.0, max_qual);
}

double GetDoubleTag(const sl::BamRecord& read, const std::string& tag) {
  uint8_t* p = bam_aux_get(read.raw(), tag.data());
  if (!p) throw std::invalid_argument("Tag does not exist");
  double result = bam_aux2f(p);
  int type = *p++;
  if (type != 'd') throw std::invalid_argument("Tag is not of double type");

  return result;
}

void AddDoubleTag(sl::BamRecord& read, const std::string& tag, double val) {
  bam_aux_append(read.raw(), tag.data(), 'd', sizeof(double), (uint8_t*)&val);
}

// Penalties adapted from svviz2
const double kGapOpen = -1.;
const double kGapExtend = -1.;

// svviz2 rescales all base qualities
double RescaleQuality(char quality, double scale = 0.25) { return scale * static_cast<double>(quality); }

double ScoreAlignment(const std::string& read_sequence, const std::string& base_qualities,
                      const std::string& ref_sequence, const sl::BamRecord& alignment) {
  int entry_read_pos = 0;
  int entry_ref_pos = alignment.PositionWithSClips();
  double log_prob = 0;  // log10(P(data|alignment))

  sl::Cigar cigar = alignment.GetCigar();
  for (const auto& cigar_entry : cigar) {
    int entry_read_end = entry_read_pos + cigar_entry.Length();
    switch (cigar_entry.Type()) {  // MIDNSHPX
      default:
        throw std::invalid_argument("CIGAR entry not implemented");
      case 'S':
        // TODO: Don't penalize shorter soft-clip regions (reduce penalty for <
        // 10 bases)
        for (; entry_read_pos < entry_read_end; entry_read_pos++, entry_ref_pos++) {
          log_prob += PhredToLogProb(RescaleQuality(base_qualities[entry_read_pos]));
        }
        break;
      case 'M':
        for (; entry_read_pos < entry_read_end; entry_read_pos++, entry_ref_pos++) {
          if (read_sequence[entry_read_pos] == ref_sequence[entry_ref_pos]) {
            auto quality = RescaleQuality(base_qualities[entry_read_pos]);
            log_prob += log10(1. - PhredToProb(quality));
          } else {
            log_prob += PhredToLogProb(RescaleQuality(base_qualities[entry_read_pos]));
          }
        }
        break;
      case 'I':
        log_prob += PhredToLogProb(RescaleQuality(base_qualities[entry_read_pos++]), kGapOpen);
        for (; entry_read_pos < entry_read_end; entry_read_pos++) {
          log_prob += PhredToLogProb(RescaleQuality(base_qualities[entry_read_pos]), kGapExtend);
        }
        break;
      case 'D':
        log_prob += kGapOpen;
        if (cigar_entry.Length() > 1) log_prob += (cigar_entry.Length() - 1) * kGapExtend;
        entry_ref_pos += cigar_entry.Length();
        break;
    }
  }

  return log_prob;
}

}  // namespace

namespace npsv2 {

double InsertSizeDistribution::operator()(int insert_size) const {
  auto entry = density_.find(insert_size);
  if (entry != density_.end()) {
    return entry->second;
  } else {
    // https://stackoverflow.com/a/10848293
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (insert_size - mean_) / std_;
    return inv_sqrt_2pi / std_ * std::exp(-0.5 * a * a);
  }
}

RealignedReadPair::RealignedReadPair(const sl::BamRecord& first) : left_(&first), right_(nullptr) {
  score_ = GetDoubleTag(*left_, "as");
}

RealignedReadPair::RealignedReadPair(const sl::BamRecord& first, const sl::BamRecord& second,
                                     const InsertSizeDistribution& insert_dist)
    : left_(&first), right_(&second), score_(0.) {
  if (left_->Position() > right_->Position()) {
    std::swap(left_, right_);
  }
  // Scoring algorithm adapted from svviz2:
  // https://github.com/nspies/svviz2/blob/44f7bfc75bf84c1db4563d9fd30bf20967d1c825/src/svviz2/io/readstatistics.py
  score_ += GetDoubleTag(*left_, "as");
  score_ += GetDoubleTag(*right_, "as");

  if (!Concordant()) {
    score_ -= 10.;
    return;
  }
  auto insert_size_prob = insert_dist(InsertSize());
  if (insert_size_prob == 0.) {
    score_ -= 10.;
    return;
  }
  score_ += log10(insert_size_prob);
}

int32_t RealignedReadPair::InsertSize() const {
  return right_->PositionWithSClips() + right_->Length() - left_->PositionWithSClips();
}

bool RealignedReadPair::Concordant() const {
  if (left_->ChrID() != right_->ChrID()) return false;

  // TODO: Check orientation

  return true;
}

std::ostream& operator<<(std::ostream& os, const RealignedReadPair& pair) {
  if (pair.left_) os << *pair.left_ << std::endl;
  if (pair.right_) os << *pair.right_ << std::endl;
  return os << pair.score_ << std::endl;
}

namespace {

void RealignRead(const IndexedSequence& index, const sl::BamRecord& read, sl::BamRecordVector& alignments,
                 int quality_offset) {
  const std::string read_seq(read.Sequence());
  const std::string base_qualities(read.Qualities(quality_offset));
  const std::string& ref_seq(index.Sequence());

  index.AlignSequence(read.Qname(), read_seq, alignments);

  for (auto& alignment : alignments) {
    auto log_prob = ScoreAlignment(read_seq, base_qualities, ref_seq, alignment);
    AddDoubleTag(alignment, "as", log_prob);
  }
}

}  // namespace

RealignedFragment::RealignedFragment(const sl::BamRecord& read1, const sl::BamRecord& read2,
                                     const IndexedSequence& index, const InsertSizeDistribution& insert_dist,
                                     int quality_offset)
    : total_log_prob_(std::numeric_limits<score_type>::lowest()) {
  pyassert(!read1.isEmpty(), "Fragment needs to include at least on read");
  RealignRead(index, read1, read1_alignments_, quality_offset);

  if (!read2.isEmpty()) {
    RealignRead(index, read2, read2_alignments_, quality_offset);
  }

  // Construct and score possible alignment pairs from these individual
  if (!read1_alignments_.empty() && !read2_alignments_.empty()) {
    for (auto& align1 : read1_alignments_) {
      for (auto& align2 : read2_alignments_) {
        read_pairs_.emplace_back(align1, align2, insert_dist);
      }
    }
  }

  // Previous NPSV only considered actual pairs, but incorporating
  // singletons slightly reduced accuracy
  // else {
  //   // Handle situation with singleton reads
  //   for (auto& align : read1_alignments_)
  //     read_pairs_.emplace_back(align);
  //   for (auto& align : read2_alignments_)
  //     read_pairs_.emplace_back(align);
  // }

  // Sort alignments in descending order by score
  std::sort(read_pairs_.begin(), read_pairs_.end(), std::greater<>());

  for (const auto& pair : read_pairs_) {
    total_log_prob_ = LogSumPow(total_log_prob_, pair.Score());
  }
}

IndexedSequence::IndexedSequence(const sl::UnalignedSequence& sequence) { Initialize(sequence); }

void IndexedSequence::Initialize(const sl::UnalignedSequence& sequence) {
  pyassert(!IsInitialized(), "BWA should not previously have been initialized");
  sequence_ = sequence;
  bwa_.ConstructIndex({sequence});
}

void IndexedSequence::AlignSequence(const std::string& name, const std::string& seq,
                                    sl::BamRecordVector& alignments) const {
  bwa_.AlignSequence(seq, name, alignments, false, 0.9, 10);
}

void IndexedSequence::AlignSequence(const sl::BamRecord& read, sl::BamRecordVector& alignments) const {
  AlignSequence(read.Qname(), read.Sequence(), alignments);
}

FragmentRealigner::FragmentRealigner(const std::string& fasta_path, double insert_size_mean, double insert_size_std)
    : insert_size_dist_(insert_size_mean, insert_size_std) {
  // Release the GIL while executing the C++ realignment code
  py::gil_scoped_release release;
  
  // Load alleles from a FASTA file
  sl::FastqReader contigs(fasta_path);
  sl::UnalignedSequence next_sequence;

  // We assumed the first sequence is the reference sequence
  pyassert(contigs.GetNextSequence(next_sequence), "Reference sequence not present in the FASTA");
  ref_index_.Initialize(next_sequence);

  // The remaining sequences at the alternate sequences
  while (contigs.GetNextSequence(next_sequence)) {
    alt_indexes_.emplace_back(next_sequence);
  }
}

std::map<std::string, double> FragmentRealigner::RealignReadPair(const std::string& name, const std::string& read1_seq,
                                                                 const std::string& read1_qual, py::kwargs kwargs) { 
  int offset = 0;
  if (kwargs && kwargs.contains("offset")) {
    offset = py::cast<int>(kwargs["offset"]);
  }

  sl::BamRecord read1, read2;
  read1.init();
  read1.SetQname(name);
  read1.SetSequence(read1_seq);
  read1.SetQualities(read1_qual, offset);

  if (kwargs && kwargs.contains("read2_seq") && kwargs.contains("read2_qual")) {
    read2.init();
    read2.SetQname(name);
    read2.SetSequence(py::cast<std::string>(kwargs["read2_seq"]));
    read2.SetQualities(py::cast<std::string>(kwargs["read2_qual"]), offset);
  }

  // Release the GIL while executing the C++ realignment code. This seems to need to be after
  // any interactions with Python objects (e.g. kwargs) 
  py::gil_scoped_release release;

  // Realign the fragment to the reference allele
  RealignedFragment ref_realignment(read1, read2, ref_index_, insert_size_dist_);
  auto total_log_prob = ref_realignment.TotalLogProb();

  std::vector<RealignedFragment> alt_realignments;
  for (int i = 0; i < NumAltAlleles(); i++) {
    // Realign the fragment to this alternate allele
    alt_realignments.emplace_back(read1, read2, alt_indexes_[i], insert_size_dist_);
    total_log_prob = LogSumPow(total_log_prob, alt_realignments.back().TotalLogProb());
  }

  RealignedFragment::score_type ref_quality = 0;
  if (ref_realignment.HasBestPair()) {
    ref_quality = LogProbToPhredQual(ref_realignment.BestPairLogProb() - total_log_prob, 40);
  }

  RealignedFragment::score_type max_alt_quality = 0;
  for (const auto& alt_realignment : alt_realignments) {
    if (alt_realignment.HasBestPair()) {
      max_alt_quality =
          std::max(max_alt_quality, LogProbToPhredQual(alt_realignment.BestPairLogProb() - total_log_prob, 40));
    }
  }

  std::map<std::string, double> results;
  results["ref_quality"] = ref_quality;
  results["max_alt_quality"] = max_alt_quality;
  return results;
}

namespace test {
std::vector<double> TestScoreAlignment(const std::string& ref_seq, const std::string& aln_path) {
  // Open the input BAM/SAM/CRAM
  sl::BamReader reader;
  reader.Open(aln_path);
  std::vector<double> scores;

  sl::BamRecord read;
  while (reader.GetNextRecord(read)) {
    auto log_prob = ScoreAlignment(read.Sequence(), read.Qualities(0), ref_seq, read);
    scores.push_back(log_prob);
  }

  reader.Close();
  return scores;
}

std::map<std::string, double> TestRealignReadPair(const std::string& fasta_path, const std::string& name,
                                                  const std::string& read1_seq, const std::string& read1_qual,
                                                  py::kwargs kwargs) {
  pyassert(kwargs && kwargs.contains("fragment_mean") && kwargs.contains("fragment_sd"),
           "Insert size distribution must be provided");

  FragmentRealigner realigner(fasta_path, py::cast<double>(kwargs["fragment_mean"]),
                              py::cast<double>(kwargs["fragment_sd"]));
  return realigner.RealignReadPair(name, read1_seq, read1_qual, kwargs);
}

}  // namespace test

}  // namespace npsv2