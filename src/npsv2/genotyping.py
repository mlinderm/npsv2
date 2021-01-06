import pysam
from .sample import sample_name_from_bam
from .variant import Variant
from . import images

VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])


def genotype_vcf(params, model_path: str, vcf_path: str, read_paths, output_path: str, image_shape):
    # Create genotyper model
    genotyper = models.TripletModel(image_shape, 1, model_path=model_path)

    # Map BAM files to sample IDs
    sample_to_reads = { sample_name_from_bam(read_path): read_path for read_path in read_paths }

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
        for sample_name in sample_to_reads.keys():
            dst_header.add_sample(sample_name)

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            for record in src_vcf_file:
                variant = Variant.from_pysam(record)

                dst_samples = []
                for sample, sample_bam in sample_to_reads.items():
                    
                    # TODO: Set a single replicate for genotyping
                    example = make_variant_example(params, variant, sample_bam, sample, label=None, simulate=True)
                    
                    # Convert example to features
                    features = {
                        "image": images._example_image(example),
                        "sim/images": images._example_sim_images(example)
                    }

                    # Predict genotype
                    dataset = tf.data.Dataset.from_tensors((features, None))
                    genotypes, *_  = genotyper.predict(dataset)
                    
                    dst_samples.append({"GT": [None, None]})



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
