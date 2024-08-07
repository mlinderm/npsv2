import io, itertools, logging, math, os, tempfile, typing
from functools import partial
import pysam
import pysam.bcftools as bcftools
import ray
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from .variant import Variant
from .sample import Sample
from . import images
from . import models
from .range import Range
from .utilities.vcf import allele_indices_from_genotype_field_index, genotype_field_index, genotype_field_len, index_variant_file, bcftools_format

PLOIDY=2
VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])


def coverage_over_region(input_bam, region: Range, reference, min_mapq=40, min_baseq=15, min_anchor=11):
    """Compute mean coverage, total coverage and region length for a genomic region"""
    region_length = region.length
    if region_length == 0:
        return (0., 0., region_length)
    
    depth_result = pysam.depth(  # pylint: disable=no-member
        "-Q", str(min_mapq),
        "-q", str(min_baseq),
        "-l", str(min_anchor),
        "-r", str(region),
        "--reference", reference,
        input_bam,
    )

    if len(depth_result) > 0:
        depths = np.loadtxt(io.StringIO(depth_result), dtype=int, usecols=2)
        total_coverage = np.sum(depths)
        return (total_coverage / region_length, total_coverage, region_length)
    else:
        return (0., 0., region_length)


def coverage_features(cfg: DictConfig, variant: Variant, sample: Sample, allele=1):
    """Compute DHFFC for specific sample and allele"""
    left_region = variant.left_flank_region(cfg.pileup.fetch_flank, allele=allele)  # TODO: Incorporate CIPOS and CIEND?
    right_region = variant.right_flank_region(cfg.pileup.fetch_flank, allele=allele)
    event_region = Range(left_region.contig, left_region.end, right_region.start)

    coverage, _, _ = coverage_over_region(sample.bam, event_region, cfg.reference)
    _, left_flank_bases, left_flank_length = coverage_over_region(sample.bam, left_region, cfg.reference)
    _, right_flank_bases, right_flank_length = coverage_over_region(sample.bam, right_region, cfg.reference)
    
    total_flank_bases = left_flank_bases + right_flank_bases
    total_flank_length = left_flank_length + right_flank_length
    
    if total_flank_bases > 0 and total_flank_length > 0:
        dhffc = coverage / (total_flank_bases / total_flank_length)
    else:
        dhffc = math.nan
    return { "DHFFC": dhffc }


def _allele_masks(variant: Variant) -> typing.List[typing.AbstractSet[int]]:
    """Generate a list of all possible allele masks for a variant assuming a ploidy of 2"""
    alleles = variant.alt_allele_indices
    return [set(a) for a in itertools.chain(itertools.combinations(alleles, 1), itertools.combinations(alleles, 2))]


def genotype_vcf(cfg: DictConfig, vcf_path: str, samples: typing.Dict[str,Sample], model_paths: typing.Sequence[str], output_path: str, progress_bar=False, evaluate=False, region: str=None):
    """Genotype VCF in samples using provided model(s) and other configuration parameters

    Args:
        cfg (DictConfig): Hydra/OmegaConf configuration
        vcf_path (str): Input VCF path
        samples (Dict[str,Sample]): Dictionary of sample name, Sample objects to genotype
        model_paths (Sequence[str]): CNN models to use for genotyping
        output_path (str): Output VCF path
        progress_bar (bool, optional): Show progress bar. Defaults to False.
        evaluate (bool, optional): Log concordance if genotypes included in input VCF. Defaults to False.

    Raises:
        NotImplementedError: Specified ploidy is not supported
    """
    assert cfg.simulation.replicates >= 1, "At least one replicate is required for genotyping"
    
    # Create image generator and genotyper model
    generator = hydra.utils.instantiate(cfg.generator, cfg)
    model = hydra.utils.instantiate(cfg.model, generator.image_shape[-3:], cfg.simulation.replicates, model_path=model_paths)
    
    with pysam.VariantFile(vcf_path, drop_samples=True) as src_vcf_file:

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
        dst_header.add_line('##FORMAT=<ID=DHFFC,Number=A,Type=Float,Description="Ratio between mean coverage in the event and the flanks">')
        dst_header.add_line('##FORMAT=<ID=FS,Number=A,Type=Integer,Description="Phred-scaled probability of Fisher exact test for strand bias">')
        dst_header.add_line('##FORMAT=<ID=SOR,Number=A,Type=Float,Description="Strand odds ratio">')

        ordered_samples = []
        for name, sample in samples.items():
            dst_header.add_sample(name)
            ordered_samples.append(sample)

        if evaluate:
            eval_samples = set(src_header.samples) & set(dst_header.samples)
            eval_table = pd.DataFrame.from_records(itertools.product(list(eval_samples),[False,True],[0]), columns=["SAMPLE","CONCORDANT","COUNT"], index=["SAMPLE","CONCORDANT"])

    if region:
        logging.info("Genotyping variants in region %s", region)
        region_args = Range.parse_literal(region).pysam_fetch
    else:
        region_args = {}


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
        
        with pysam.VariantFile(vcf_path, drop_samples=yield_example or not evaluate) as vcf_file:
            if not yield_example and evaluate:
                vcf_file.subset_samples(samples.keys())
            for i, record in enumerate(vcf_file.fetch(**region_args)):
                if i % num_shards != index:
                    continue
                if yield_example:
                    variant = Variant.from_pysam(record)

                    # Examples is "2-D" list of lists, i.e. list of allele mask combinations for example sample
                    allele_masks = _allele_masks(variant)
                    examples = []
                    for sample in ordered_samples:
                        sample_examples = []
                        for mask in allele_masks:
                            if len(mask) == 1:
                                # Extract additional coverage features (which depends on the allele)
                                sample_features = coverage_features(cfg, variant, sample, allele=next(iter(mask)))
                                addl_features = {
                                    "addl/DHFFC": images._float_feature([sample_features["DHFFC"]]),
                                }
                            else:
                                addl_features = {}
                            
                            example = images.make_variant_example(cfg, variant, sample.bam, sample, label=None, simulate=True, generator=generator, alleles=mask, addl_features=addl_features)
                            sample_examples.append(example)
                        examples.append(sample_examples)
                    
                    yield (i, examples)
                else:
                    yield (i, record)

    with tempfile.TemporaryDirectory() as output_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        logging.info("Initializing ray with %d threads", cfg.threads)
        # TODO: Seem to run into memory issues starting multiple clusters. Set memory based on allocation?
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=output_dir, ignore_reinit_error=True, include_dashboard=False)

        unsorted_output_path = os.path.join(output_dir, "genotypes.vcf.gz")
        with pysam.VariantFile(unsorted_output_path, mode="wz", header=dst_header) as dst_vcf_file:
            # Create parallel iterators. We use a partial wrapper because the generator alone can't be be pickled.
            # Ray gather_async ensures that examples generated in each shared are generated in order, so we
            # use a corresponding set of iterators for obtaining the associated VCF records.
            num_shards = cfg.threads
            example_it = ray.util.iter.from_iterators([partial(_vcf_shard, num_shards, i) for i in range(num_shards)])
            record_its = [_vcf_shard(num_shards, i, yield_example=False) for i in range(num_shards)]

            for index, sample_examples in tqdm(example_it.gather_async(), desc="Genotyping variants", disable=not progress_bar):
                # Obtain the corresponding VCF record
                record_index, record = next(record_its[index % num_shards])
                assert record_index == index, "Mismatch VCF variant and result from threading"
                assert len(sample_examples) == len(ordered_samples), "Mismatch in expected number of samples"
                
                variant = Variant.from_pysam(record)
                allele_masks = _allele_masks(variant)
                distance_len = genotype_field_len(variant.num_alt, PLOIDY)

                dst_samples = []
                for sample_name, mask_examples in zip(dst_header.samples, sample_examples):                    
                    assert len(mask_examples) == len(allele_masks), "Mismatch in expected number of allele combinations"
                    
                    vcf_distances = np.ones(distance_len, dtype=np.float32)
                    dhffc = [None] * variant.num_alt
                    fisher_strand = [None] * variant.num_alt
                    strand_orientation_bias = [None] * variant.num_alt
                    for mask, example in zip(allele_masks, mask_examples):
                        example_variant = images._example_variant(example)
                        assert example_variant.start == variant.start and example_variant.end == variant.end, "Mismatch VCF variant and result from threading"
                        
                        # Convert example to features
                        features = {
                            "image": images._example_image(example),
                            "sim/images": images._example_sim_images(example),
                            "variant/encoded": example_variant.SerializeToString(),
                        }

                        # Predict genotype using one or more models (by taking the mean of the distances across models and replicates)
                        dataset = tf.data.Dataset.from_tensors((features, None))
                        _, distances = model.predict(cfg, dataset)
                        distances = tf.math.reduce_mean(distances, axis=0) # Reduce multiple replicates for an SV
                        
                        # Convert distances to "genotype likelihood" ordering expected by VCF (TODO: Average reference allele values?)
                        if len(mask) == 1:
                            # The mask contains just the alternate allele
                            for i, gt in enumerate(itertools.combinations_with_replacement({0} | mask, PLOIDY)):
                                vcf_distances[genotype_field_index(gt)] = distances[i]
                            dhffc[next(iter(mask)) - 1] = images._example_addl_attribute(example, "addl/DHFFC")
                            fisher_strand[next(iter(mask)) - 1] = int(images._example_addl_attribute(example, "addl/fisher_strand"))
                            strand_orientation_bias[next(iter(mask)) - 1] = images._example_addl_attribute(example, "addl/strand_orientation_bias")
                        elif len(mask) == 2:
                            # For multiple alleles we only care about AC=2 (compound het. distance)
                            vcf_distances[genotype_field_index(mask)] = distances[2] 
                        else:
                            raise NotImplementedError("Only ploidy <= 2 is currently supported")
                        
                    # pysam checks the Python type, so we use the `list` method to convert to Python float, int, etc.
                    gt_index = np.argmin(vcf_distances)
                    dst_samples.append({
                        "GT": allele_indices_from_genotype_field_index(gt_index, variant.num_alt, PLOIDY),
                        "DS": vcf_distances.round(4).tolist(),
                        "DHFFC": np.round(dhffc, 3,).tolist(),
                        "FS": fisher_strand,
                        "SOR": np.round(strand_orientation_bias, 3).tolist(),
                    })

                    if evaluate and sample_name in eval_samples:
                        # If source VCF has genotype information, compute concordance
                        concordant = gt_index == genotype_field_index(variant.genotype_indices(sample_name))
                        eval_table.loc[(sample_name, concordant), "COUNT"] += 1

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

            if evaluate:
                conc_table = eval_table.transform(lambda x: x / x.sum()).loc[pd.IndexSlice[:,True],:].reset_index(level=1, drop=True).rename(columns={ "COUNT": "CONC"})
                for name, conc in conc_table.itertuples():
                    logging.info("Concordance for sample %s: %f", name, conc)

        # Sort output file and index (if relevant)
        bcftools.sort("-O", bcftools_format(output_path), "-o", output_path, "-T", output_dir, unsorted_output_path, catch_stdout=False)
        index_variant_file(output_path)

        # Shutdown Ray after genotyping is complete
        ray.shutdown()