import typing
from functools import partial
import pysam
import ray
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .sample import sample_name_from_bam
from .variant import Variant
from . import images
from . import models

VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])

def ac_to_genotype(ac):
    # Current assumes biallelic, diploid site
    return [0] * (2-ac) + [1] * ac


def genotype_vcf(params, model_path: str, vcf_path: str, samples, output_path: str, image_shape, progress_bar=False,):
    assert params.replicates >= 1, "At least one replicate is required for genotyping"
    
    # We currently just use ray for the CPU-side work, specifically simulating the SVs
    ray.init(num_cpus=params.threads, num_gpus=0, _temp_dir=params.tempdir, ignore_reinit_error=True, include_dashboard=False)

    # Create genotyper model
    # TODO: Extract shape from saved model
    #genotyper = models.WindowedJointEmbeddingsModel(image_shape + (images.IMAGE_CHANNELS,), params.replicates, model_path=model_path)
    genotyper = models.JointEmbeddingsModel(image_shape + (images.IMAGE_CHANNELS,), params.replicates, model_path=model_path)
    

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
        
        ordered_samples = []
        for name, sample in samples.items():
            dst_header.add_sample(name)
            ordered_samples.append(sample)

        def _vcf_shard(num_shards: int, index: int) -> typing.Iterator[tf.train.Example]:
            """Generator of tf.train.Example for use as a Ray parallel iterator

            Args:
                num_shards (int): Number of shards executing in parallel
                index (int): Worker index
            """
            # Try to reduce the number of threads TF creates since we are running multiple instances of TF via Ray
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
                for i, record in enumerate(vcf_file):
                    if i % num_shards == index:
                        variant = Variant.from_pysam(record)
                        examples = [images.make_variant_example(params, variant, sample.bam, sample, label=None, simulate=True, image_shape=image_shape, replicates=params.replicates) for sample in ordered_samples]
                        yield examples


        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            # Create parallel iterators. We use a partial wrapper because the generator alone can't be be pickled.
            it = ray.util.iter.from_iterators([partial(_vcf_shard, params.threads, i) for i in range(params.threads)])
            
            # gather_sync ensures variants are generated in order at the cost of load imbalance
            for record, examples in tqdm(zip(src_vcf_file, it.gather_sync()), desc="Genotyping variants", disable=not progress_bar):
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
                    genotypes, distances, *_  = genotyper.predict(dataset)
                    
                    # pysam checks the Python type, so we use the `list` method to convert to Python float, int, etc.
                    dst_samples.append({
                        "GT": ac_to_genotype(np.argmax(genotypes)),
                        "DS": np.squeeze(distances).round(4).tolist(),
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
