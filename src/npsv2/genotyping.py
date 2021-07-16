import io, os, tempfile, typing
from functools import partial
import pysam
import pysam.bcftools as bcftools
import ray
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import hydra
from .sample import sample_name_from_bam
from .variant import Variant
from . import images
from . import models
from .range import Range
from .utilities.vcf import index_variant_file, bcftools_format

VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])

def ac_to_genotype(ac):
    # Current assumes biallelic, diploid site
    return [0] * (2-ac) + [1] * ac

def coverage_over_region(input_bam, region: Range, reference, min_mapq=40, min_baseq=15, min_anchor=11):
    """Compute coverage"""
    depth_result = pysam.depth(  # pylint: disable=no-member
        "-Q", str(min_mapq),
        "-q", str(min_baseq),
        "-l", str(min_anchor),
        "-r", str(region),
        "--reference", reference,
        input_bam,
    )
    
    # start, end are 0-indexed half-open coordinates
    region_length = region.length
    if len(depth_result) > 0 and region_length > 0:
        depths = np.loadtxt(io.StringIO(depth_result), dtype=int, usecols=2)
        total_coverage = np.sum(depths)
        return (total_coverage / region_length, total_coverage, region_length)
    else:
        return (0., 0., region_length)


def genotype_vcf(cfg, vcf_path: str, samples, output_path: str, progress_bar=False,):
    assert cfg.simulation.replicates >= 1, "At least one replicate is required for genotyping"
    
    # We currently just use ray for the CPU-side work, specifically simulating the SVs
    ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=tempfile.gettempdir(), ignore_reinit_error=True, include_dashboard=False)

    # Create image generator and genotyper model
    generator = hydra.utils.instantiate(cfg.generator, cfg)
    model = hydra.utils.instantiate(cfg.model, generator.image_shape[-3:], 1)

    with tempfile.TemporaryDirectory() as output_dir, pysam.VariantFile(vcf_path, drop_samples=True) as src_vcf_file:

        # Create header for destination file
        src_header = src_vcf_file.header
        dst_header = pysam.VariantHeader()

        # Copy existing header fields
        for record in src_header.records:
            if record.type in VCF_HEADER_TYPES_TO_COPY:
                dst_header.add_record(record)

        # Add NPSV2-specific header lines
        # TODO: Add metadata about the model, etc.
        dst_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        dst_header.add_line('##FORMAT=<ID=DS,Number=G,Type=Float,Description="Distance between real and simulated data">')
        dst_header.add_line('##FORMAT=<ID=DHFFC,Number=1,Type=Float,Description="Ratio between mean coverage in the event and the flanks">')
        

        ordered_samples = []
        for name, sample in samples.items():
            dst_header.add_sample(name)
            ordered_samples.append(sample)

        def _vcf_shard(num_shards: int, index: int, yield_example: bool = True):
            """Generator of tf.train.Example for use as a Ray parallel iterator

            Args:
                num_shards (int): Number of shards executing in parallel
                index (int): Worker index
            """
            try:
                # Try to reduce the number of threads TF creates since we are running multiple instances of TF via Ray
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except RuntimeError:
                pass
            
            with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
                for i, record in enumerate(vcf_file):
                    if i % num_shards == index:
                        if yield_example:
                            variant = Variant.from_pysam(record)
                            examples = [images.make_variant_example(cfg, variant, sample.bam, sample, label=None, simulate=True, generator=generator) for sample in ordered_samples]
                            
                            # Generate other features as well
                            left_region = variant.left_flank_region(cfg.pileup.fetch_flank)  # TODO: Incorporate CIPOS and CIEND?
                            event_region = variant.reference_region
                            right_region = variant.right_flank_region(cfg.pileup.fetch_flank)
                            
                            coverage, _, _ = coverage_over_region(sample.bam, event_region, cfg.reference)
                            _, left_flank_bases, left_flank_length = coverage_over_region(sample.bam, left_region, cfg.reference)
                            _, right_flank_bases, right_flank_length = coverage_over_region(sample.bam, right_region, cfg.reference)
                            
                            total_flank_bases = left_flank_bases + right_flank_bases
                            total_flank_length = left_flank_length + right_flank_length
                            
                            if total_flank_bases > 0 and total_flank_length > 0:
                                DHFFC = coverage / (total_flank_bases / total_flank_length)
                            else:
                                DHFFC = 1. if coverage > 0 else 0.
                            yield (i, examples, DHFFC)
                        else:
                            yield (i, record)

        unsorted_output_path = os.path.join(output_dir, "genotypes.vcf.gz")
        with pysam.VariantFile(unsorted_output_path, mode="w", header=dst_header) as dst_vcf_file:
            # Create parallel iterators. We use a partial wrapper because the generator alone can't be be pickled.
            # Ray gather_async ensures that examples generated in each shared are generated in order, so we
            # use a corresponding set of iterators for obtaining the associated VCF records.
            num_shards = cfg.threads
            example_it = ray.util.iter.from_iterators([partial(_vcf_shard, num_shards, i) for i in range(num_shards)])
            record_its = [_vcf_shard(num_shards, i, yield_example=False) for i in range(num_shards)]

            for index, examples, DHFFC in tqdm(example_it.gather_async(), desc="Genotyping variants", disable=not progress_bar):
                # Obtain the corresponding VCF record
                record_index, record = next(record_its[index % num_shards])
                assert record_index == index, "Mismatch VCF variant and result from threading"
                variant = Variant.from_pysam(record)

                dst_samples = []
                for example in examples:                    
                    example_variant = images._example_variant(example)
                    assert example_variant.start == variant.start and example_variant.end == variant.end, "Mismatch VCF variant and result from threading"
                    
                    # Convert example to features
                    features = {
                        "image": images._example_image(example),
                        "sim/images": images._example_sim_images(example)
                    }

                    # Predict genotype
                    dataset = tf.data.Dataset.from_tensors((features, None))
                    _, distances, *_  = model.predict(cfg, dataset)
                    #print(distances)
                    distances = tf.math.reduce_mean(distances, axis=0) # Reduce multiple replicates for an SV
                    genotypes = tf.nn.softmax(-distances)
                    #print(distances, tf.math.reduce_mean(distances, axis=0), tf.nn.softmax(-tf.math.reduce_mean(distances, axis=0)))
                    
                    # pysam checks the Python type, so we use the `list` method to convert to Python float, int, etc.
                    dst_samples.append({
                        "GT": ac_to_genotype(np.argmax(genotypes)),
                        "DS": np.squeeze(distances).round(4).tolist(),
                        "DHFFC": DHFFC,
                    })

                # Create and write new record with genotypes
                dst_record = dst_header.new_record(
                    contig=record.contig,
                    start=record.start,
                    stop=record.stop,
                    alleles=record.alleles,
                    id=record.id,
                    qual=record.qual,
                    filter=record.filter,
                    info=record.info,
                    samples=dst_samples,
                )
                dst_vcf_file.write(dst_record)

        # Sort output file and index (if relevant)
        bcftools.sort("-O", bcftools_format(output_path), "-o", output_path, "-T", output_dir, unsorted_output_path, catch_stdout=False)
        index_variant_file(output_path)