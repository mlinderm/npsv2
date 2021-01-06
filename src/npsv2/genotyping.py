import pysam
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
    # Create genotyper model
    # TODO: Extract shape from saved model
    genotyper = models.TripletModel(image_shape + (images.IMAGE_CHANNELS,), 1, model_path=model_path)

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
        dst_header.add_line("##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>")
        dst_header.add_line("##FORMAT=<ID=DS,Number=G,Type=Float,Description='Distance between real and simulated data'>")
        
        ordered_samples = []
        for name, sample in samples.items():
            dst_header.add_sample(name)
            ordered_samples.append(sample)

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            for record in tqdm(src_vcf_file, desc="Genotyping variants", disable=not progress_bar):
                variant = Variant.from_pysam(record)

                dst_samples = []
                for sample in ordered_samples:
                    # TODO: Set the minimum number of replicates for genotyping here
                    example = images.make_variant_example(params, variant, sample.bam, sample, label=None, simulate=True)
                    
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
