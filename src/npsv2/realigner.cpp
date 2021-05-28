#include "realigner.hpp"

#include "SeqLib/FastqReader.h"
#include "utility.hpp"

namespace {

double LogSumPow(double acc, double prob) {
  double m = std::max(acc, prob);
  return m + log10(pow(10.,acc - m) + pow(10.,prob - m));
}

double PhredToProb(double phred) { return pow(10.0, phred / -10.0); }

double PhredToLogProb(double quality, double penalty = 0.) { return (-quality / 10.) + penalty; }

double LogProbToPhredQual(double prob, double max_qual) {
  if (prob == 0.)
    return max_qual;
  else
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

bool IUPACMatch(char base, char ref) {
  switch (ref) {
    default:
      throw std::invalid_argument(std::string("Invalid reference base: ") + ref);
    case 'A':
    case 'C':
    case 'G':
    case 'T':
      return base == ref;
    case 'R':	return base == 'A' || base == 'G';
    case 'Y':	return base == 'C' || base == 'T';
    case 'S':	return base == 'G' || base == 'C';
    case 'W':	return base == 'A' || base == 'T';
    case 'K':	return base == 'G' || base == 'T';
    case 'M':	return base == 'A' || base == 'C';
    case 'B':	return base == 'C' || base == 'G' || base == 'T';
    case 'D':	return base == 'A' || base == 'G' || base == 'T';
    case 'H':	return base == 'A' || base == 'C' || base == 'T';
    case 'V':	return base == 'A' || base == 'C' || base == 'G';
    case 'N':	return true;
  }
}

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
          if (IUPACMatch(read_sequence[entry_read_pos], ref_sequence[entry_ref_pos])) {
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

double MaxScoreAlignment(const std::string& read_sequence, const std::string& base_qualities) {
  double log_prob = 0;  // log10(P(data|alignment))
  for (int i=0; i < read_sequence.size(); i++) {
    auto quality = RescaleQuality(base_qualities[i]);
    log_prob += log10(1. - PhredToProb(quality));
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

double InsertSizeDistribution::Max() const {
  static const double inv_sqrt_2pi = 0.3989422804014327;
  return inv_sqrt_2pi / std_;
}

RealignedReadPair::RealignedReadPair(const sl::BamRecord& first) : left_(&first), right_(nullptr) {
  score_ = GetDoubleTag(*left_, "as");
  max_score_ = GetDoubleTag(*left_, "ms");
}

RealignedReadPair::RealignedReadPair(const sl::BamRecord& first, const sl::BamRecord& second,
                                     const InsertSizeDistribution& insert_dist)
    : left_(&first), right_(&second), score_(0.), max_score_(0.) {
  if (left_->Position() > right_->Position()) {
    std::swap(left_, right_);
  }
  // Scoring algorithm adapted from svviz2:
  // https://github.com/nspies/svviz2/blob/44f7bfc75bf84c1db4563d9fd30bf20967d1c825/src/svviz2/io/readstatistics.py
  score_ += GetDoubleTag(*left_, "as");
  score_ += GetDoubleTag(*right_, "as");

  max_score_ += GetDoubleTag(*left_, "ms");
  max_score_ += GetDoubleTag(*right_, "ms");

  if (!Concordant()) {
    score_ -= 10.;
    max_score_ -= 10.;
    return;
  }
  auto insert_size_prob = insert_dist(InsertSize());
  if (insert_size_prob == 0.) {
    score_ -= 10.;
    max_score_ -= 10.;
    return;
  }
  score_ += log10(insert_size_prob);
  max_score_ += log10(insert_dist.Max());
}

namespace {
  sl::GenomicRegion ReadRegion(const sl::BamRecord& read) {
    // The read region is 0-indexed with a exclusive end coordinate, but SeqLib's operations on GenomicRegions
    // assume 1-indexed regions with inclusive start and end coordinates. So we convert those coordinates here.
    return sl::GenomicRegion(read.ChrID(), read.Position() + 1, read.PositionEnd());
  }
}

sl::GenomicRegion RealignedReadPair::FragmentRegion() const {
  if (left_ && right_ && left_->ChrID() == right_->ChrID()) {
    if (left_->ChrID() == right_->ChrID()) {
      return sl::GenomicRegion(left_->ChrID(), std::min(left_->Position() + 1, right_->Position()) + 1,
                               std::max(left_->PositionEnd(), right_->PositionEnd()));
    }
  } else if (left_) {
    return ReadRegion(*left_);
  } else if (right_) {
    return ReadRegion(*right_);
  }

  return sl::GenomicRegion();
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
  if (pair.left_) os << *pair.left_ << " as:" << GetDoubleTag(*pair.left_, "as") << std::endl;
  if (pair.right_) os << *pair.right_ << " as:" << GetDoubleTag(*pair.right_, "as") << std::endl;
  return os << pair.score_ << std::endl;
}

namespace {

void RealignRead(const IndexedSequence& index, const sl::BamRecord& read, sl::BamRecordVector& alignments,
                 int quality_offset) {
  const std::string read_seq(read.Sequence());
  const std::string base_qualities(read.Qualities(quality_offset));
  const std::string& ref_seq(index.IUPACSequence());

  index.AlignSequence(read.Qname(), read_seq, alignments);
  auto max_log_prob = MaxScoreAlignment(read_seq, base_qualities);
  for (auto& alignment : alignments) {
    auto log_prob = ScoreAlignment(read_seq, base_qualities, ref_seq, alignment);
    AddDoubleTag(alignment, "as", log_prob);
    AddDoubleTag(alignment, "ms", max_log_prob);
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

  // Construct and score possible alignment pairs
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

const std::string& IndexedSequence::IUPACSequence() const {
  if (!iupac_sequence_.Seq.empty())
    return iupac_sequence_.Seq;
  else 
    return sequence_.Seq;
}

void IndexedSequence::AlignSequence(const std::string& name, const std::string& seq,
                                    sl::BamRecordVector& alignments) const {
  bwa_.AlignSequence(seq, name, alignments, false, 0.9, 10);
}

void IndexedSequence::AlignSequence(const sl::BamRecord& read, sl::BamRecordVector& alignments) const {
  AlignSequence(read.Qname(), read.Sequence(), alignments);
}

namespace {
  sl::GenomicRegion BreakpointToGenomicRegion(const std::string& region, const sl::BamHeader& header) {
      return (region.empty()) ? sl::GenomicRegion() : sl::GenomicRegion(region, header);
  }
}

FragmentRealigner::FragmentRealigner(const std::string& fasta_path, const BreakpointList& breakpoints, double insert_size_mean, double insert_size_std, py::kwargs kwargs)
    : insert_size_dist_(insert_size_mean, insert_size_std) {
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
  
  // Convert the list of breakpoint strings into GenomicRegions
  pyassert(NumAltAlleles() == breakpoints.size(), "Inconsistent number of alt sequences and breakpoints");
  for (int i=0; i<NumAltAlleles(); i++) {
    const auto & allele_breakpoints = breakpoints[i];
    breakpoints_.push_back({
      BreakpointToGenomicRegion(std::get<0>(allele_breakpoints), RefHeader()),
      BreakpointToGenomicRegion(std::get<1>(allele_breakpoints), RefHeader()), 
      BreakpointToGenomicRegion(std::get<2>(allele_breakpoints), AltHeader(i)),
      BreakpointToGenomicRegion(std::get<3>(allele_breakpoints), AltHeader(i)) 
    });
  }

  // Load the FASTA file with IUPAC sequence if provided
  if (kwargs && kwargs.contains("iupac_fasta_path")) {
    sl::FastqReader contigs(py::cast<std::string>(kwargs["iupac_fasta_path"]));
    // We assumed the first sequence is the reference sequence
    pyassert(contigs.GetNextSequence(next_sequence), "Reference sequence not present in the IUPAC FASTA");
    ref_index_.SetIUPACSequence(next_sequence);
    for (int i = 0; i < NumAltAlleles(); i++) {
      pyassert(contigs.GetNextSequence(next_sequence), "Missing alternate sequence in the IUPAC FASTA");
      alt_indexes_[i].SetIUPACSequence(next_sequence);
    }
  }
}

namespace {
  std::string ToString(const sl::GenomicRegion& region, const sl::BamHeader& header) {
    return region.ChrName(header) + ":" + std::to_string(region.pos1) + "-" + std::to_string(region.pos2);
  }
}

FragmentRealigner::RealignTuple FragmentRealigner::RealignReadPair(const std::string& name, const std::string& read1_seq,
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

  // if (read1.Qname() == "HISEQ1:18:H8VC6ADXX:2:1111:14992:77166") {
  //   std::cerr << ref_realignment.BestPair() << std::endl;
  // }

  std::vector<RealignedFragment> alt_realignments;
  for (int i = 0; i < NumAltAlleles(); i++) {
    // Realign the fragment to this alternate allele
    alt_realignments.emplace_back(read1, read2, alt_indexes_[i], insert_size_dist_);
    total_log_prob = LogSumPow(total_log_prob, alt_realignments.back().TotalLogProb());
  }

  RealignedFragment::score_type ref_score = 0, ref_quality = 0, ref_max_score = 0;
  bool ref_breakpoint_overlap = false;
  if (ref_realignment.HasBestPair()) {
    auto & best_pair = ref_realignment.BestPair();
    ref_score = best_pair.Score();
    ref_max_score = best_pair.MaxPossibleScore();
    ref_quality = LogProbToPhredQual(ref_score - total_log_prob, 40);
    //std::cerr << "Ref: " << best_pair.Score() << " " << best_pair.MaxPossibleScore() << std::endl;
    auto fragment_region = best_pair.FragmentRegion();
    for (int i = 0; i < NumAltAlleles(); i++) {
      ref_breakpoint_overlap |= fragment_region.GetOverlap(breakpoints_[i][0]) == GenomicRegionOverlap::ContainsArg;
      ref_breakpoint_overlap |= fragment_region.GetOverlap(breakpoints_[i][1]) == GenomicRegionOverlap::ContainsArg;
    }
  }

  RealignedFragment::score_type max_alt_score = 0, max_alt_quality = 0, max_alt_max_score = 0;
  bool max_alt_breakpoint_overlap = false;
  for (int i = 0; i < NumAltAlleles(); i++) {
    const auto& alt_realignment = alt_realignments[i];
    if (alt_realignment.HasBestPair()) {
      auto & best_pair = alt_realignment.BestPair();
      auto alt_score = best_pair.Score();
      auto alt_quality = LogProbToPhredQual(alt_score - total_log_prob, 40);
      if (alt_quality >= max_alt_quality) {
        max_alt_score = alt_score;
        max_alt_quality = alt_quality;
        max_alt_max_score = best_pair.MaxPossibleScore();
        
        auto fragment_region = best_pair.FragmentRegion();
        max_alt_breakpoint_overlap = 
          (fragment_region.GetOverlap(breakpoints_[i][2]) == GenomicRegionOverlap::ContainsArg) ||
          (fragment_region.GetOverlap(breakpoints_[i][3]) == GenomicRegionOverlap::ContainsArg);
      }
      // if (read1.Qname() == "HISEQ1:18:H8VC6ADXX:2:1111:14992:77166") {
      //   std::cerr << alt_realignment.BestPair() << std::endl;
      // }
    }
  }
  // if (ref_quality > (max_alt_quality + 1))
  //   std::cerr << "Ref: " << total_log_prob << std::endl;
  // else if (max_alt_quality > (ref_quality + 1))
  //   std::cerr << "Alt: " << total_log_prob << std::endl;
  return std::make_tuple(ref_quality, ref_breakpoint_overlap, ref_score, ref_max_score,
                         max_alt_quality, max_alt_breakpoint_overlap, max_alt_score, max_alt_max_score);
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

FragmentRealigner::RealignTuple TestRealignReadPair(const std::string& fasta_path,
                                                           const FragmentRealigner::BreakpointList& breakpoints,
                                                           const std::string& name, const std::string& read1_seq,
                                                           const std::string& read1_qual, py::kwargs kwargs) {
  pyassert(kwargs && kwargs.contains("fragment_mean") && kwargs.contains("fragment_sd"),
           "Insert size distribution must be provided");

  FragmentRealigner realigner(fasta_path, breakpoints, py::cast<double>(kwargs["fragment_mean"]),
                              py::cast<double>(kwargs["fragment_sd"]), kwargs);
  return realigner.RealignReadPair(name, read1_seq, read1_qual, kwargs);
}

}  // namespace test

}  // namespace npsv2