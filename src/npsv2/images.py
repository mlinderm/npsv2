import functools, logging, os, random, subprocess, sys, tempfile
import numpy as np
import pysam
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import ray

from .variant import Variant
from .range import Range
from .pileup import Pileup, FragmentTracker, AlleleAssignment, BaseAlignment, ReadPileup
from . import npsv2_pb2
from .realigner import FragmentRealigner, realign_fragment
from .simulation import RandomVariants, simulate_variant_sequencing, augment_samples
from .sample import Sample
from .util.config import Config, merge_config

IMAGE_HEIGHT = 50 #100
IMAGE_WIDTH = 300
IMAGE_CHANNELS = 5
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

PADDING = 100

MAX_PIXEL_VALUE = 254.0  # Adapted from DeepVariant

BASE_CHANNEL = 0
ALIGNED_BASE_PIXEL = 255
SOFT_CLIP_BASE_PIXEL = 128

ALIGNED_TO_PIXEL = {
    BaseAlignment.ALIGNED: ALIGNED_BASE_PIXEL,
    BaseAlignment.SOFT_CLIP: SOFT_CLIP_BASE_PIXEL,
}

REF_INSERT_SIZE_CHANNEL = 1
ALT_INSERT_SIZE_CHANNEL = 2
INSERT_SIZE_MEAN_PIXEL = 128
INSERT_SIZE_SD_PIXEL = 24

ALLELE_CHANNEL = 3
AMB_PIXEL = 50
REF_PIXEL = 150
ALT_PIXEL = 250

ALLELE_TO_PIXEL = {
    AlleleAssignment.AMB: AMB_PIXEL,
    AlleleAssignment.REF: REF_PIXEL,
    AlleleAssignment.ALT: ALT_PIXEL,
}

MAPQ_CHANNEL = 4
MAX_MAPQ = 60

VARIANT_BAND_HEIGHT = 5 # Adapted from DeepVariant
VARIANT_MAPQ = 60

class ImageConfig(Config):
    # TODO: Move parameters over to more flexible config classes
    
    # Reference band is 5 pixels in DeepVariant
    variant_band_height: int = 0 

    assign_delta: float = 1.0

    window_size: int = 50
    flank_windows: int = 1


def _fragment_zscore(sample: Sample, fragment_length: int):
    return (fragment_length - sample.mean_insert_size) / sample.std_insert_size


def _create_realigner(params, variant, sample: Sample):
    fasta_path, _, _ = variant.synth_fasta(reference_fasta=params.reference, dir=params.tempdir, flank=params.flank)
    return FragmentRealigner(fasta_path, sample.mean_insert_size, sample.std_insert_size)


def _add_variant_strip(config: ImageConfig, sample: Sample, variant: Variant, pileup: Pileup, image_tensor):
    image_tensor[:config.variant_band_height, :, BASE_CHANNEL] = ALIGNED_TO_PIXEL[BaseAlignment.ALIGNED]
    image_tensor[:config.variant_band_height, :, MAPQ_CHANNEL] = min(VARIANT_MAPQ / MAX_MAPQ, 1.0) * MAX_PIXEL_VALUE
    
    ref_zscore, alt_zscore = abs(variant.length_change() / sample.std_insert_size), 0
    image_tensor[:config.variant_band_height, :, REF_INSERT_SIZE_CHANNEL] = np.clip(
        INSERT_SIZE_MEAN_PIXEL + ref_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
    )
    image_tensor[:config.variant_band_height, :, ALT_INSERT_SIZE_CHANNEL] = np.clip(
        INSERT_SIZE_MEAN_PIXEL + alt_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
    )

    # Clip out the region containing the variant
    for col_slice in pileup.region_columns(variant.reference_region):
        image_tensor[:config.variant_band_height, col_slice, :] = 0


def create_single_example(params, variant, read_path, region, sample: Sample, image_shape=None, realigner=None, **kwargs):
    config = merge_config(ImageConfig(), kwargs)
    
    if isinstance(region, str):
        region = Range.parse_literal(region)

    if image_shape and len(image_shape) != 2:
        raise ValueError("Image shape must be (rows, height) sequence")
    elif image_shape:
        tensor_shape = (image_shape[0], region.length, IMAGE_CHANNELS)
    else:
        tensor_shape = (IMAGE_HEIGHT, region.length, IMAGE_CHANNELS)

    image_tensor = np.zeros(tensor_shape, dtype=np.uint8)

    if realigner is None:
        realigner = _create_realigner(params, variant, sample)

    fragments = FragmentTracker()

    with pysam.AlignmentFile(read_path) as alignment_file:
        # Expand query region to capture straddling reads
        for read in alignment_file.fetch(
            contig=region.contig, start=region.start - params.flank, stop=region.end + params.flank
        ):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            fragments.add_read(read)

    # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
    pileup = Pileup(region)
    
    realigned_reads = []
    straddling_fragments = []

    left_region = variant.left_flank_region(params.flank)  # TODO: Incorporate CIPOS and CIEND?
    right_region = variant.right_flank_region(params.flank)

    for fragment in fragments:
        # At present we render reads based on the original alignment so we only realign fragments that could overlap
        # the image window
        if fragment.reads_overlap(region):
            allele, ref_meta, alt_meta = realign_fragment(realigner, fragment, assign_delta=config.assign_delta)
            realigned_reads.append((allele, fragment.read1, ref_meta, alt_meta))
            realigned_reads.append((allele, fragment.read2, ref_meta, alt_meta))
        else:
            allele = AlleleAssignment.AMB

        # Only record the zscore for reads that straddle the event
        if fragment.fragment_straddles(left_region, right_region, min_aligned=3):
            ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
            straddling_fragments.append((fragment, ref_zscore, alt_zscore))
        else:
            ref_zscore = None
            alt_zscore = None

        pileup.add_fragment(fragment, allele=allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

    # Add variant strip at the top of the image, clipping out the variant region
    if config.variant_band_height > 0:
        _add_variant_strip(config, sample, variant, pileup, image_tensor)

    # Add pileup bases to the image
    read_pixels = tensor_shape[0] - config.variant_band_height
    for j, column in enumerate(pileup):
        for i, base in enumerate(column.ordered_bases(order="aligned", max_bases=read_pixels), start=config.variant_band_height):
            image_tensor[i, j, BASE_CHANNEL] = ALIGNED_TO_PIXEL[base.aligned]
            image_tensor[i, j, MAPQ_CHANNEL] = min(base.mapq / MAX_MAPQ, 1.0) * MAX_PIXEL_VALUE
            #image_tensor[i, j, ALLELE_CHANNEL] = ALLELE_TO_PIXEL[base.allele]
           
            # if base.ref_zscore is not None:
            #     image_tensor[i, j, REF_INSERT_SIZE_CHANNEL] = np.clip(
            #         INSERT_SIZE_MEAN_PIXEL + base.ref_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
            #     )
            # if base.alt_zscore is not None:
            #     image_tensor[i, j, ALT_INSERT_SIZE_CHANNEL] = np.clip(
            #         INSERT_SIZE_MEAN_PIXEL + base.alt_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
            #     )

    # Render realigned reads as "full" reads, not as pileup "bases"
    def _filter_reads(realigned_read):
        allele, read, (ref_qual, _), (alt_qual, _) = realigned_read
        # Keep reads in the image window that overlap one of the variant breakpoints
        return read and allele != AlleleAssignment.AMB and region.get_overlap(read) > 0
    
    realigned_reads = [read[:2] for read in realigned_reads if _filter_reads(read)]
    if len(realigned_reads) > read_pixels:
        realigned_reads = random.sample(realigned_reads, k=read_pixels)
    realigned_reads.sort(key=lambda x: x[1].reference_start)

    for i, (allele, read) in enumerate(realigned_reads, start=config.variant_band_height):
        _, col_slices = pileup.read_columns(read)
        for col_slice, _ in col_slices:
            image_tensor[i, col_slice, ALLELE_CHANNEL] = ALLELE_TO_PIXEL[allele]

    # Render fragment insert size as "bars" since straddling reads may not be in the image
    if len(straddling_fragments) > read_pixels:
        straddling_fragments = random.sample(straddling_fragments, k=read_pixels)
    straddling_fragments.sort(key = lambda x: x[0].fragment_start)
    for i, (fragment, ref_zscore, alt_zscore) in enumerate(straddling_fragments, start=config.variant_band_height):
        for col_slice in pileup.region_columns(fragment.fragment_region):
            image_tensor[i, col_slice, REF_INSERT_SIZE_CHANNEL] = np.clip(
                INSERT_SIZE_MEAN_PIXEL + ref_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
            )
            image_tensor[i, col_slice, ALT_INSERT_SIZE_CHANNEL] = np.clip(
                INSERT_SIZE_MEAN_PIXEL + alt_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
            )

    # Create consistent image size
    if image_shape and image_tensor.shape[:2] != image_shape:
        # resize converts to float directly (however convert_image_dtype assumes floats are in [0-1]) so
        # we use cast instead
        image_tensor = tf.cast(tf.image.resize(image_tensor, image_shape), dtype=tf.uint8).numpy()

    return image_tensor


def create_windowed_example(params, variant, read_path, regions, sample: Sample, realigner=None, **kwargs):
    config = merge_config(ImageConfig(), kwargs)

    tensor_shape = (len(regions), IMAGE_HEIGHT, regions[0].length, IMAGE_CHANNELS)
    image_tensor = np.zeros(tensor_shape, dtype=np.uint8)
    for region in regions[1:]:
        assert region.length == regions[0].length, "All regions must have identical lengths"

    if realigner is None:
        realigner = _create_realigner(params, variant, sample)

    # Determine the outer extent of all windows
    region = functools.reduce(lambda a, b: a.union(b), regions)

    fragments = FragmentTracker()
    
    with pysam.AlignmentFile(read_path) as alignment_file:
        # Expand query region to capture straddling reads
        for read in alignment_file.fetch(
            contig=region.contig, start=region.start - params.flank, stop=region.end + params.flank
        ):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            fragments.add_read(read)

    pileup = ReadPileup(regions)

    for fragment in fragments:
        # At present we render reads based on the original alignment so skip any read that doesn't map to the image
        # region
        if not fragment.reads_overlap(region):
            continue
        
        # Realign reads
        allele, ref_meta, alt_meta = realign_fragment(realigner, fragment, assign_delta=config.assign_delta)

        # Determine the Z-score for all reads 
        ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
        alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
       
        pileup.add_fragment(fragment, allele=allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

    read_pixels = tensor_shape[1]
    for w, window_region in enumerate(regions):
        # Get and potential downsample reads overlapping this window
        window_reads = pileup.overlapping_reads(window_region, max_reads=read_pixels)
        for i, read in enumerate(window_reads):
            for col_slice, aligned in pileup.read_columns(window_region, read):
                image_tensor[w, i, col_slice, ALLELE_CHANNEL] = ALLELE_TO_PIXEL[read.allele]
                image_tensor[w, i, col_slice, BASE_CHANNEL] = ALIGNED_TO_PIXEL[aligned]
                image_tensor[w, i, col_slice, MAPQ_CHANNEL] = min(read.mapq / MAX_MAPQ, 1.0) * MAX_PIXEL_VALUE
                image_tensor[w, i, col_slice, REF_INSERT_SIZE_CHANNEL] = np.clip(
                    INSERT_SIZE_MEAN_PIXEL + read.ref_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
                )
                image_tensor[w, i, col_slice, ALT_INSERT_SIZE_CHANNEL] = np.clip(
                    INSERT_SIZE_MEAN_PIXEL + read.alt_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE
                )

    return image_tensor


def _genotype_to_label(genotype, alleles={1}):
    # TODO: Handle no-calls
    count = 0
    for gt in genotype:
        if gt == -1:
            return None
        elif gt in alleles:
            count += 1
    return count


# Adapted from DeepVariant
def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a list of string / byte."""
    if isinstance(list_of_strings, type(tf.constant(0))):
        list_of_strings = [list_of_strings.numpy()]  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))


def _int_feature(list_of_ints):
    """Returns a int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def make_variant_example(params, variant: Variant, read_path: str, sample: Sample, label=None, simulate=False, replicates=0, **kwargs):
    config = merge_config(ImageConfig(), kwargs)
    
    # Construct realigner once for all images for this variant
    realigner = _create_realigner(params, variant, sample)

    # Construct image for "real" data
    if params.single_image:
        # Construct a single "padded" region to render as the pileup image
        variant_region = variant.reference_region
        padding = max((IMAGE_WIDTH - variant_region.length) // 2, PADDING)
        example_region = variant_region.expand(padding)

        image_tensor = create_single_example(
            params, variant, read_path, example_region, sample, realigner=realigner, **kwargs
        )
    else:
        # Constructing a windowed image by centering breakpoints in a window and tiling the interior of the event
        example_regions = variant.window_regions(config.window_size, config.flank_windows)
        image_tensor = create_windowed_example(
            params, variant, read_path, example_regions, sample, realigner=realigner, **kwargs
        )


    feature = {
        "variant/encoded": _bytes_feature([variant.as_proto().SerializeToString()]),
        "image/shape": _int_feature(image_tensor.shape),
        "image/encoded": _bytes_feature(tf.io.serialize_tensor(image_tensor)),
    }

    if label is not None:
        feature["label"] = _int_feature([label])

    if simulate and replicates > 0:
        # A 5/6-D tensor for simulated images (AC, REPLICATES) + (WINDOWS?, ROW, COLS, CHANNELS)
        feature["sim/images/shape"] = _int_feature((3, replicates) + image_tensor.shape)

        # Generate synthetic training images
        ac_encoded_images = [None] * 3
    
        # If we are augmenting the simulated data, use the provided statistics for the first example, so it
        # will hopefully be similar to the real data and the augment the remaining replicates
        if params.augment:
            repl_samples = augment_samples(sample, replicates, keep_original=True)
        else:
            repl_samples = [sample] * replicates

        for allele_count in range(3):
            with tempfile.TemporaryDirectory(dir=params.tempdir) as tempdir:
                # Generate the FASTA file for this zygosity
                fasta_path, ref_contig, alt_contig = variant.synth_fasta(
                    reference_fasta=params.reference, ac=allele_count, flank=params.flank, dir=tempdir
                )

                # Generate and image synthetic bam files
                repl_encoded_images = []
                
                sim_replicates = replicates if allele_count != 0 or not params.sample_ref else 0         
                for i in range(sim_replicates):
                    try:
                        replicate_bam_path = simulate_variant_sequencing(
                            params, fasta_path, allele_count, repl_samples[i], dir=tempdir
                        )
                    except ValueError:
                        logging.error("Failed to synthesize data for %s with AC=%d", str(variant), allele_count)
                        raise

                    if params.single_image:
                        synth_image_tensor = create_single_example(
                            params,
                            variant,
                            replicate_bam_path,
                            example_region,
                            repl_samples[i],
                            realigner=realigner,
                            **kwargs
                        )
                    else:
                        synth_image_tensor = create_windowed_example(
                            params,
                            variant,
                            replicate_bam_path,
                            example_regions,
                            repl_samples[i],
                            realigner=realigner,
                            **kwargs
                        )
 
                    repl_encoded_images.append(synth_image_tensor)

                # Fill remaining images with sampled reference variants
                if allele_count == 0 and params.sample_ref:
                    for random_variant in random_variants.generate(variant, replicates - sim_replicates):
                        if params.single_image:
                            random_variant_region = random_variant.reference_region.expand(padding)
                            synth_image_tensor = create_single_example(
                                params, random_variant, read_path, random_variant_region, sample, **kwargs
                            )
                        else:
                            random_variant_regions = random_variant.window_regions(config.window_size, config.flank_windows)
                            synth_image_tensor =  create_windowed_example(
                                params, random_variant, read_path, random_variant_regions, sample, **kwargs
                            )
                        repl_encoded_images.append(synth_image_tensor)

                # Stack all of the image replicates into a tensor
                ac_encoded_images[allele_count] = np.stack(repl_encoded_images)

        # Stack the replicated images for the 3 genotypes (0/0, 0/1, 1/1) into tensor
        sim_image_tensor = np.stack(ac_encoded_images)
        feature[f"sim/images/encoded"] = _bytes_feature(tf.io.serialize_tensor(sim_image_tensor))

    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_vcf_examples(
    params,
    vcf_path: str,
    read_path: str,
    sample: Sample,
    sample_or_label=None,
    region: str = None,
    **kwargs,
):
    with pysam.VariantFile(vcf_path) as vcf_file:
        # Prepare function to extract genotype label
        if sample_or_label is not None and isinstance(sample_or_label, str):
            samples = vcf_file.header.samples
            sample_index = next((i for i, s in enumerate(samples) if s == sample_or_label), -1,)
            if sample_index == -1:
                raise ValueError("Sample identifier is not present in the file")
            logging.info("Using %s genotypes as labels (VCF sample index %d)", sample_or_label, sample_index)
            vcf_file.subset_samples([sample_or_label])
            label_extractor = lambda variant: _genotype_to_label(variant.genotype_indices(sample_index))
        else:
            logging.info("Using fixed AC=%d as label", sample_or_label)
            vcf_file.subset_samples([])  # Drop all samples
            label_extractor = lambda variant: sample_or_label

        # Prepare random variant generator (if specified)
        if params.sample_ref:
            random_variants = RandomVariants(params.reference, params.exclude_bed)

        if region:
            query_range = Range.parse_literal(region)
            variant_iter = vcf_file.fetch(**query_range.pysam_fetch)
        else:
            variant_iter = vcf_file

        for record in variant_iter:
            variant = Variant.from_pysam(record)
            assert variant.is_biallelic(), "Multi-allelic sites not yet supported"

            # To avoid duplicated entries, only generate images for variants that start within region
            if region and not query_range.contains(variant.start):
                continue

            label = label_extractor(variant)
            yield make_variant_example(params, variant, read_path, sample,  label=label, replicates=params.replicates, **kwargs)
           


def _filename_to_compression(filename: str) -> str:
    if filename.endswith(".gz"):
        return "GZIP"
    else:
        return None


def _chunk_genome(ref_path: str, vcf_path: str, chunk_size=30000000):
    regions = []
    with pysam.FastaFile(ref_path) as ref_file, pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
        for contig, length in zip(ref_file.references, ref_file.lengths):
            if vcf_file.get_tid(contig) != -1:
                for start in range(1, length, chunk_size):
                    regions.append(f"{contig}:{start}-{min(start + chunk_size, length)}")
    return regions


def _ray_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def vcf_to_tfrecords(
    params,
    vcf_path: str,
    read_path: str,
    output_path: str,
    sample: Sample,
    image_shape=None,
    sample_or_label=None,
    simulate=False,
    progress_bar=False,
    **kwargs,
):

    # Unfortunately we can't use TF's built-in multithreading because of the extensive use of Python (the GIL serializes the execution). Instead we use
    # Ray and generate a separate file for each VCF chunk that are then merged at the end.
    ray.init(num_cpus=params.threads, num_gpus=0, _temp_dir=params.tempdir, ignore_reinit_error=True, include_dashboard=False)

    @ray.remote
    def _chunk_to_tfrecords(output_path: str, region: str = None):
        # Try to reduce the number of threads TF creates since we are running multiple instances of TF via Ray
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Generate records for each region into a file
        all_examples = make_vcf_examples(
            params,
            vcf_path,
            read_path,
            sample,
            sample_or_label=sample_or_label,
            region=region,
            image_shape=image_shape,
            simulate=simulate,
            **kwargs,
        )
        with tf.io.TFRecordWriter(output_path, _filename_to_compression(output_path)) as file_writer:
            total_variants = 0
            for example in all_examples:
                file_writer.write(example.SerializeToString())
                total_variants += 1
            return output_path, total_variants

    with tempfile.TemporaryDirectory(dir=params.tempdir) as tempdir:
        # Create tasks for each genomic chunk
        regions = _chunk_genome(params.reference, vcf_path)
        ray_tasks = [
            _chunk_to_tfrecords.remote(os.path.join(tempdir, f"chunk{i}.tfrecords.gz"), region)
            for i, region in enumerate(regions)
        ]

        chunk_files, total_variants = [], 0
        with tqdm(desc="Generating variant records (in chunks)", disable=not progress_bar) as pb:
            for chunk_file, variants in _ray_iterator(ray_tasks):
                pb.update(variants)
                if variants > 0:
                    chunk_files.append(chunk_file)
                    total_variants += variants

        # Merge the per-chunk dataset files into one
        merged_dataset = tf.data.TFRecordDataset(
            chunk_files, compression_type="GZIP", num_parallel_reads=params.threads
        )
        with tf.io.TFRecordWriter(output_path, _filename_to_compression(output_path)) as file_writer:
            for serialized_example in tqdm(
                merged_dataset,
                desc="Merging chunks into TFRecords file",
                total=total_variants,
                disable=not progress_bar,
            ):
                file_writer.write(serialized_example.numpy())


def _features_variant(features):
    return npsv2_pb2.StructuralVariant.FromString(features["variant/encoded"].numpy())


def _example_variant(example):
    encoded_variant = example.features.feature["variant/encoded"].bytes_list.value[0]
    return npsv2_pb2.StructuralVariant.FromString(encoded_variant)


def _example_image(example):
    image_data = tf.io.parse_tensor(example.features.feature["image/encoded"].bytes_list.value[0], tf.uint8).numpy()
    return image_data


def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value)


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _example_sim_images_shape(example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value)
    else:
        return (3, 0, None, None, None)


def _example_sim_images(example):
    image_data = tf.io.parse_tensor(
        example.features.feature["sim/images/encoded"].bytes_list.value[0], tf.uint8
    ).numpy()
    return image_data


def _extract_metadata_from_first_example(filename):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())

    image_shape = _example_image_shape(example)
    ac, replicates, *sim_image_shape = _example_sim_images_shape(example)
    if replicates > 0:
        assert ac == 3, "Incorrect number of genotypes in simulated data"
        assert image_shape == tuple(sim_image_shape), "Simulated and actual image shapes don't match"

    return image_shape, replicates


def _flatten_image(image_tensor):
    # TODO: Combine all the channels into a single image, perhaps BASE, INSERT_SIZE, ALLELE (with
    # mapq as alpha)...
    channels = [BASE_CHANNEL, REF_INSERT_SIZE_CHANNEL, ALLELE_CHANNEL]
    return Image.fromarray(image_tensor[:, :, channels], mode="RGB")    


def _tensor_to_image(image_tensor):
    shape = image_tensor.shape
    if len(shape) == 3:
        return _flatten_image(image_tensor)
    elif len(shape) == 4:
        windows, height, width, _ = shape
        composite_image = Image.new("RGB", (windows*width, height))
        for i in range(windows):
            sub_image = _flatten_image(image_tensor[i])
            composite_image.paste(sub_image, (i*width, 0))
        return composite_image
    else:
        assert False


def example_to_image(example: tf.train.Example, out_path: str, with_simulations=False, margin=10, max_replicates=1):
    # Adapted from DeepVariant
    features = example.features.feature

    image_tensor = _example_image(example)
    real_image = _tensor_to_image(image_tensor)

    _, replicates, *_ = _example_sim_images_shape(example)
    if with_simulations and replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode,  (width + 2 * (width + margin), height + replicates * (height + margin)))
        image.paste(real_image, (width + margin, 0))

        synth_tensor = _example_sim_images(example)
        for ac in range(3):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[ac, repl]
                synth_image = _tensor_to_image(synth_image_tensor)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def load_example_dataset(filenames, with_label=False, with_simulations=False) -> tf.data.Dataset:
    if isinstance(filenames, str):
        filenames = [filenames]
    assert len(filenames) > 0

    # Extract image shape from the first example
    shape, replicates = _extract_metadata_from_first_example(filenames[0])

    proto_features = {
        "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
    }
    if with_label:
        proto_features["label"] = tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)
    if with_simulations and replicates > 0:
        proto_features.update(
            {
                "sim/images/shape": tf.io.FixedLenFeature(shape=(len(shape) + 2,), dtype=tf.int64),
                "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            }
        )

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            "variant/encoded": parsed_features["variant/encoded"],
            "image": tf.io.parse_tensor(parsed_features["image/encoded"], tf.uint8),
        }
        if with_simulations:
            features["sim/images"] = tf.io.parse_tensor(parsed_features["sim/images/encoded"], tf.uint8)

        if with_label:
            return features, parsed_features["label"]
        else:
            return features, None

    return tf.data.TFRecordDataset(filenames=filenames, compression_type=_filename_to_compression(filenames[0])).map(
        map_func=_process_input
    )

