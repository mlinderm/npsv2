import pysam
import hydra
from hydra.experimental import compose, initialize
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from npsv2.sample import Sample
from npsv2.variant import Variant
from npsv2.images import _realigner
from npsv2.pileup import FragmentTracker, AlleleAssignment
from npsv2.realigner import FragmentRealigner, realign_fragment, AlleleRealignment

VCF_PATH="tests/data/1_67808460_67808624_DEL.vcf.gz"
BAM_PATH="tests/data/1_67806460_67811624.bam"
PLOT_PATH="plot.pdf"

initialize(config_path="../src/npsv2/conf")
cfg = compose(config_name="config", overrides=[
    "reference=/data/human_g1k_v37.fasta",
])

sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)
        

generator = hydra.utils.instantiate(cfg.generator, cfg=cfg)
with pysam.VariantFile(VCF_PATH) as vcf_file, pysam.AlignmentFile(BAM_PATH, "rb") as bam_file:
    rows = []
    for record in vcf_file:
        variant = Variant.from_pysam(record)
        if not variant.is_biallelic():
            continue

        realigner = _realigner(variant, sample, reference=cfg.reference, flank=cfg.pileup.realigner_flank)

        region = generator.image_regions(variant)
        fetch_region = region.expand(cfg.pileup.fetch_flank)

        fragments = FragmentTracker()
        for read in bam_file.fetch(contig=fetch_region.contig, start=fetch_region.start, stop=fetch_region.end):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            fragments.add_read(read)

        realigned_reads = []
        for fragment in fragments:
            if fragment.reads_overlap(region):
                realignment = realign_fragment(realigner, fragment, assign_delta=cfg.pileup.assign_delta)
                realigned_reads.append((fragment.read1, realignment))
                realigned_reads.append((fragment.read2, realignment))

        def _filter_reads(realigned_read):
            read, realignment = realigned_read
            # Keep reads in the image window that overlap one of the variant breakpoints
            return read and realignment.allele == AlleleAssignment.ALT and realignment.breakpoint and region.get_overlap(read) > 0
        
        
        for read, realignment in filter(_filter_reads, realigned_reads):
            rows.append(pd.DataFrame({
                "SVLEN": variant.length_change(),
                "FILTER": variant._record.filter,
                "NORM_SCORE": realignment.normalized_score, 
            }))
        
    table = pd.concat(rows, ignore_index=True)
    print(table)
    sns.histplot(data=table, x="NORM_SCORE", binwidth=1)
    plt.savefig(PLOT_PATH)