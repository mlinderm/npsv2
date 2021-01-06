import json
import pysam

_SAMPLE_STATS_FIELDS = ("sequencer", "read_length", "mean_coverage", "mean_insert_size", "std_insert_size")

def sample_name_from_bam(bam_path) -> str:
    """Extract sample name from BAM"""
    with pysam.AlignmentFile(bam_path) as bam:
        read_groups = bam.header["RG"]
        samples = set([rg["SM"] for rg in read_groups])
        assert len(samples) == 1, f"BAM file {bam.filename} must contain a single sample"
        sample, = samples  # Extract single value from set
        return sample


class Sample:
    def __init__(self, name, **kwargs):
        self.name = name
        
        self.bam = kwargs.get("bam", None)
        self.gender = kwargs.get("gender", 0)  # Use PED encoding
        
        # Statistics fields initialized to None
        for k in _SAMPLE_STATS_FIELDS:
            setattr(self, k, kwargs.get(k, None))

        self._gc_normalized_coverage = kwargs.get("gc_normalized_coverage", {})
        

    def gc_normalized_coverage(self, gc_fraction: int) -> float:
        return self._gc_normalized_coverage.get(gc_fraction, 1.0)

    @classmethod
    def from_json(cls, json_path: str, min_gc_bin=100, max_gc_error=0.01) -> "Sample":
        with open(json_path, "r") as file:
            sample_info = json.load(file)

            fields = {k: sample_info[k] for k in _SAMPLE_STATS_FIELDS}

            # Optional fields
            fields["bam"] = sample_info.get("bam", None)

            # Filter GC entries with limited data
            gc_normalized_coverage = {}
            for gc, norm_covg in sample_info["gc_normalized_coverage"].items():
                if (
                    sample_info["gc_bin_count"].get(gc, 0) >= min_gc_bin
                    and sample_info["gc_normalized_coverage_error"].get(gc, 0) <= max_gc_error
                ):
                    gc_normalized_coverage[round(float(gc) * 100)] = norm_covg
            fields["gc_normalized_coverage"] = gc_normalized_coverage

            return cls(sample_info["sample"], **fields)