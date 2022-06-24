import json, logging, io, os, subprocess, tempfile, typing
import pysam
from shlex import quote
import numpy as np
import pandas as pd
import pybedtools.bedtool as bed

_SAMPLE_STATS_FIELDS = ("sequencer", "read_length", "mean_coverage", "mean_insert_size", "std_insert_size")


def sample_name_from_bam(bam_path) -> str:
    """Extract sample name from BAM"""
    with pysam.AlignmentFile(bam_path) as bam:
        read_groups = bam.header["RG"]
        samples = set([rg["SM"] for rg in read_groups])
        assert len(samples) == 1, f"BAM file {bam.filename} must contain a single sample"
        (sample,) = samples  # Extract single value from set
        return sample


class Sample:
    def __init__(self, name, **kwargs):
        self.name = name

        self.bam = kwargs.get("bam", None)
        self.gender = kwargs.get("gender", 0)  # Use PED encoding

        # Statistics fields initialized to None
        for k in _SAMPLE_STATS_FIELDS:
            setattr(self, k, kwargs.get(k, None))

        self._chrom_normalized_coverage = kwargs.get("chrom_normalized_coverage", {})
        self._gc_normalized_coverage = kwargs.get("gc_normalized_coverage", {})

    def gc_normalized_coverage(self, gc_fraction: int) -> float:
        return self._gc_normalized_coverage.get(gc_fraction, 1.0)

    def chrom_mean_coverage(self, chrom: str) -> float:
        """Return mean coverage for specific chromosome

        Args:
            chrom (str): Chromosome

        Returns:
            float: Mean coverage
        """
        return self._chrom_normalized_coverage.get(chrom, 1.0) * self.mean_coverage

    @classmethod
    def from_json(cls, json_path: str, min_gc_bin=100, max_gc_error=0.01) -> "Sample":
        with open(json_path, "r") as file:
            sample_info = json.load(file)

            fields = {k: sample_info[k] for k in _SAMPLE_STATS_FIELDS}

            # Optional fields
            fields["bam"] = sample_info.get("bam", None)
            fields["chrom_normalized_coverage"] = sample_info.get("chrom_normalized_coverage", {})

            # Filter GC entries with limited data
            gc_normalized_coverage = {}
            for gc, norm_covg in sample_info["gc_normalized_coverage"].items():
                if (
                    sample_info.get("gc_bin_count", {}).get(gc, 0) >= min_gc_bin
                    and sample_info.get("gc_normalized_coverage_error", {}).get(gc, 0) <= max_gc_error
                ):
                    gc_normalized_coverage[round(float(gc) * 100)] = norm_covg
            fields["gc_normalized_coverage"] = gc_normalized_coverage

            return cls(sample_info["sample"], **fields)


def _compute_coverage_with_samtools(
    read_path: str, fasta_path: str, covg_regions=5000, depth_samples=200, min_samples_per_chrom=5, threads=1
) -> typing.Tuple[float, dict, dict, dict]:
    """Compute coverage across windows with samtools (run via goleft)

    Args:
        read_path (str): Read file (BAM or CRAM)
        fasta_path (str): Reference fasta
        covg_regions (int, optional): Split genome in buckets of equal amounts of data . Defaults to 5000.
        depth_samples (int, optional): Sample regions to determine coverage. Defaults to 200.
        min_samples_per_chrom (int, optional): Minimum sampled regions for each chromosome. Defaults to 5.
        threads (int, optional): Number of threads. Defaults to 1.

    Returns:
        typing.Tuple[float,dict,dict,dict]: Mean coverage, dictionaries of chromosome and GC-normalized coverage, count of each GC bin
    """
    with tempfile.TemporaryDirectory() as output_dir:
        indexsplit_commandline = f"goleft \
            indexsplit \
            --n {covg_regions} \
            --fai {quote(fasta_path + '.fai')} \
           {read_path + '.crai' if read_path.endswith('cram') else read_path}"
        indexsplit = subprocess.check_output(indexsplit_commandline, shell=True, universal_newlines=True)
        indexsplit_table = pd.read_csv(
            io.StringIO(indexsplit), sep="\t", names=["chrom", "start", "end", "sum", "split"]
        )

        # Drop entries outside main chromosomes and randomly sample remaining regions, ensuring similar number of regions from each chromosome
        indexsplit_table = indexsplit_table[
            indexsplit_table.chrom.str.contains("^(?:chr)?(?:\d{1,2}|[XY])$", regex=True)
        ]

        if depth_samples < covg_regions:
            depth_regions = indexsplit_table.sample(depth_samples)

            # Make sure we have a minimum number of regions for each chrom
            indexsplit_group = indexsplit_table.groupby("chrom")

            def _min_sample(table):
                return (
                    indexsplit_group.get_group(table.name).sample(min_samples_per_chrom)
                    if table.shape[0] < min_samples_per_chrom
                    else table
                )

            depth_regions = depth_regions.groupby("chrom").apply(_min_sample).reset_index(0, drop=True)
        else:
            depth_regions = indexsplit_table

        regions_file = os.path.join(output_dir, "regions.bed")
        depth_regions.to_csv(regions_file, columns=["chrom", "start", "end"], header=False, index=False, sep="\t")

        prefix = os.path.join(output_dir, "depth")
        depth_commandline = f"goleft \
            depth \
            --processes {threads} \
            --stats \
            --reference {quote(fasta_path)} \
            --prefix {prefix} \
            --bed {regions_file} \
           {read_path}"

        subprocess.check_call(
            depth_commandline,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        depth_table = pd.read_csv(
            prefix + ".depth.bed",
            sep="\t",
            names=["chrom", "start", "end", "mean", "GC", "CpG", "Masked"],
            dtype={"chrom": str},
        )
        depth_table["len"] = depth_table.end - depth_table.start

        def _mean_coverage_group(table):
            return np.sum(table["len"] * table["mean"]) / np.sum(table["len"])

        # Global mean coverage
        mean_coverage = _mean_coverage_group(depth_table)

        # Compute normalized per-chromosome coverage
        chrom_norm_covg = (depth_table.groupby("chrom").apply(_mean_coverage_group) / mean_coverage).to_dict()

        # Compute GC normalized coverage (using only the autosome)
        autosome_depth_table = depth_table[depth_table.chrom.str.contains("^(?:chr)?(?:\d{1,2})$", regex=True)]
        gc_bin = np.int64(autosome_depth_table.GC.round(2) * 100)

        gc_norm_covg = (autosome_depth_table.groupby(gc_bin).apply(_mean_coverage_group) / mean_coverage).to_dict()
        gc_norm_covg_count = autosome_depth_table.groupby(gc_bin).size().to_dict()

        return mean_coverage, chrom_norm_covg, gc_norm_covg, gc_norm_covg_count


def _compute_coverage_with_indexcov(read_path: str, fasta_path: str) -> dict:
    # Generate GC and chromosome normalized coverage for the entire BAM file
    with tempfile.TemporaryDirectory() as output_dir:
        prefix = os.path.basename(output_dir)

        indexcov_commandline = f"goleft \
            indexcov \
            --extranormalize \
            --directory {output_dir} \
            --fai {quote(fasta_path + '.fai')} \
            {quote(read_path + '.crai' if read_path.endswith('cram') else read_path)}"

        subprocess.check_call(
            indexcov_commandline,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Compute chromosome and GC normalized coverage
        windows_table = (
            # pylint: disable=unexpected-keyword-arg
            bed.BedTool(fn=os.path.join(output_dir, prefix + "-indexcov.bed.gz"))
            .nucleotide_content(fi=fasta_path)
            .to_dataframe(
                index_col=False,
                header=0,
                usecols=[0, 1, 2, 3, 7, 8, 10, 11, 12],
                names=[
                    "chrom",
                    "start",
                    "end",
                    "norm_covg",
                    "num_C",
                    "num_G",
                    "num_N",
                    "num_oth",
                    "seq_len",
                ],
                dtype={"chrom": str},
            )
        )
        if windows_table.shape[0] == 0:
            return {}

        # Remove windows with no alignable data
        windows_table["align_len"] = windows_table.seq_len - windows_table.num_N - windows_table.num_oth
        windows_table = windows_table[windows_table.align_len != 0]

        def norm_coverage_group(table):
            weights = table.align_len / np.sum(table.align_len)
            return np.sum(weights * table.norm_covg)

        norm_coverage_by_chrom = windows_table.groupby("chrom").apply(norm_coverage_group).to_dict()

        return norm_coverage_by_chrom


def compute_read_stats(cfg, read_path: str) -> dict:
    # Generate stats for the entire read file using goleft covstats
    logging.info("Computing coverage and insert size statistics with goleft")
    covstats_commandline = f"goleft \
        covstats \
        --fasta {quote(cfg.reference)} \
        {read_path}"

    covstats = subprocess.check_output(covstats_commandline, shell=True, universal_newlines=True)
    covstats_table = pd.read_csv(io.StringIO(covstats), sep="\t")
    (covstats_record,) = covstats_table[covstats_table.bam == read_path].to_dict("records")

    # Compute the chromosomal and GC normalized coverage. Use all the regions to get
    # more consistent depth estimates.
    logging.info("Computing normalized coverage with parallelized samtools")
    mean_coverage, chrom_norm_covg, gc_norm_covg, gc_norm_covg_count = _compute_coverage_with_samtools(
        read_path, cfg.reference, covg_regions=5000, depth_samples=5000, min_samples_per_chrom=0, threads=cfg.threads
    )

    # Construct stats dictionary that can be written to JSON
    stats = {
        "sample": covstats_record["sample"],
        "sequencer": cfg.sequencer,
        "bam": read_path,
        "read_length": covstats_record["read_length"],
        "mean_insert_size": covstats_record["template_mean"],
        "std_insert_size": covstats_record["template_sd"],
        "mean_coverage": mean_coverage,
        "chrom_normalized_coverage": chrom_norm_covg,
        "gc_normalized_coverage": gc_norm_covg,
        "gc_bin_count": gc_norm_covg_count,
    }
    return stats
