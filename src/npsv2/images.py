import logging, os, random, subprocess, sys, tempfile
import numpy as np
import pysam
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from .variant import Variant
from .range import Range
from .pileup import Pileup, FragmentTracker, AlleleAssignment, BaseAlignment
from . import npsv2_pb2
from .realigner import FragmentRealigner, realign_fragment 
from .simulation import RandomVariants, simulate_variant_sequencing, augment_samples
from .sample import Sample

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 300
IMAGE_CHANNELS = 5
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

PADDING = 100

MAX_PIXEL_VALUE = 254 # Adapted from DeepVariant

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
REF_PIXEL = 128
ALT_PIXEL = 255

ALLELE_TO_PIXEL = {
    AlleleAssignment.AMB: 0,
    AlleleAssignment.REF: REF_PIXEL,
    AlleleAssignment.ALT: ALT_PIXEL,
}

MAPQ_CHANNEL = 4
MAX_MAPQ = 60


def _fragment_zscore(sample: Sample, fragment_length: int):
    return (fragment_length - sample.mean_insert_size) / sample.std_insert_size


def _create_realigner(params, variant, sample: Sample):
    fasta_path, _, _ = variant.synth_fasta(reference_fasta=params.reference, dir=params.tempdir, flank=params.flank)
    return FragmentRealigner(fasta_path, sample.mean_insert_size, sample.std_insert_size)


def create_single_example(params, variant, read_path, region, sample: Sample, image_shape=None, realigner=None):
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

    #pileup = Pileup(region)
    fragments = FragmentTracker()

    with pysam.AlignmentFile(read_path) as alignment_file:
        # Expand query region to capture straddling reads (TODO: Make flank a parameter)
        for read in alignment_file.fetch(
            contig=region.contig, start=region.start - params.flank, stop=region.end + params.flank
        ):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            #pileup.add_read(read)
            fragments.add_read(read)
    
    # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
    pileup = Pileup(region)
    
    left_region = variant.left_flank_region(params.flank)  # TODO: Incorporate CI
    right_region = variant.right_flank_region(params.flank)

    for fragment in fragments:
        # At present we render reads based on the original alignment so we only realign fragments that could overlap
        # the image window
        if fragment.reads_overlap(region):
            allele, _, _ = realign_fragment(realigner, fragment, assign_delta=1.0)
        else:
            allele = AlleleAssignment.AMB    
        
        # Only record the zscore for reads that straddle the event
        if fragment.fragment_straddles(left_region, right_region, min_aligned=3):
            ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
        else:
            ref_zscore = None
            alt_zscore = None

        pileup.add_fragment(fragment, allele=allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)
        

    for j, column in enumerate(pileup):
        for i, base in enumerate(column.ordered_bases(max_bases=tensor_shape[0])):
            image_tensor[i, j, BASE_CHANNEL] = ALIGNED_TO_PIXEL[base.aligned]
            image_tensor[i, j, ALLELE_CHANNEL] = ALLELE_TO_PIXEL[base.allele]
            image_tensor[i, j, MAPQ_CHANNEL] = min(base.mapq / MAX_MAPQ, 1.0) * MAX_PIXEL_VALUE
            if base.ref_zscore is not None:
                image_tensor[i, j, REF_INSERT_SIZE_CHANNEL] = np.clip(INSERT_SIZE_MEAN_PIXEL + base.ref_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE)
            if base.alt_zscore is not None:
                image_tensor[i, j, ALT_INSERT_SIZE_CHANNEL] = np.clip(INSERT_SIZE_MEAN_PIXEL + base.alt_zscore * INSERT_SIZE_SD_PIXEL, 1, MAX_PIXEL_VALUE)
      

    # left_region = variant.left_flank_region(params.flank)  # TODO: Incorporate CI
    # right_region = variant.right_flank_region(params.flank)

    # realigned_reads = []

    # irow = 0
    # for fragment in fragments:
    #     if irow < tensor_shape[0] and fragment.fragment_straddles(left_region, right_region, min_aligned=3):
    #         ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
    #         image_tensor[irow, :, REF_INSERT_SIZE_CHANNEL] = INSERT_SIZE_MEAN_PIXEL + ref_zscore * INSERT_SIZE_SD_PIXEL
    #         alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
    #         image_tensor[irow, :, ALT_INSERT_SIZE_CHANNEL] = INSERT_SIZE_MEAN_PIXEL + alt_zscore * INSERT_SIZE_SD_PIXEL
    #         irow += 1

   


    # Create consistent image size
    if image_shape and image_tensor.shape[:2] != image_shape:
        # resize converts to float, directly (however convert_image_dtype assumes floats are in [0-1]) so
        # we use cast instead
        image_tensor = tf.cast(tf.image.resize(image_tensor, image_shape), dtype=tf.uint8).numpy()

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


def make_vcf_examples(
    params, vcf_path: str, read_path: str, sample: Sample, image_shape=None, sample_or_label=None, simulate=False, region: str=None,
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

            # TODO: Handle odd sized variants when padding
            variant_region = variant.reference_region
            padding = max((IMAGE_WIDTH - variant_region.length) // 2, PADDING)
            example_region = variant_region.expand(padding)

            # Construct realigner once for all images for this variant
            realigner = _create_realigner(params, variant, sample)

            # Construct image for "real" data
            image_tensor = create_single_example(
                params, variant, read_path, example_region, sample, image_shape=image_shape, realigner=realigner
            )
            feature = {
                "variant/encoded": _bytes_feature([variant.as_proto().SerializeToString()]),
                "image/shape": _int_feature(image_tensor.shape),
                "image/encoded": _bytes_feature(tf.io.serialize_tensor(image_tensor)),
            }

            label = label_extractor(variant)
            if label is not None:
                feature["label"] = _int_feature([label])

            if simulate and params.replicates > 0:
                # A 5-D tensor for simulated images (AC, REPLICATES, ROW, COLS, CHANNELS)
                feature["sim/images/shape"] = _int_feature((3, params.replicates) + image_tensor.shape)

                # Generate synthetic training images
                ac_encoded_images = [None] * 3
                if params.sample_ref:
                    # Sample random variants from the genome to create the 0/0 replicates
                    ac_to_sim = (1, 2)
                
                    repl_encoded_images = []
                    for random_variant in random_variants.generate(variant, n=params.replicates):
                        random_variant_region = random_variant.reference_region.expand(padding)
                        synth_image_tensor = create_single_example(
                            params,
                            random_variant,
                            read_path,
                            random_variant_region,
                            sample,
                            image_shape=image_shape,
                        )
                        repl_encoded_images.append(synth_image_tensor)

                    # Stack all of the image replicates into 4-D tensor (REPLICATES, ROW, COLS, CHANNELS)
                    ac_encoded_images[0] = np.stack(repl_encoded_images)
                else:
                    ac_to_sim = (0, 1, 2)

                # If we are augmenting the simulated data, use the provided statistics for the first example, so it
                # will hopefully be similar to the real data and the augment the remaining replicates
                if params.augment:
                    repl_samples = augment_samples(sample, params.replicates, keep_original=True)
                else:
                    repl_samples = [sample] * params.replicates

                for allele_count in ac_to_sim:
                    with tempfile.TemporaryDirectory(dir=params.tempdir) as tempdir:
                        # Generate the FASTA file for this zygosity
                        fasta_path, ref_contig, alt_contig = variant.synth_fasta(
                            reference_fasta=params.reference, ac=allele_count, flank=params.flank, dir=tempdir
                        )

                        # Generate and image synthetic bam files
                        repl_encoded_images = []
                        for i in range(params.replicates):
                            replicate_bam_path = simulate_variant_sequencing(params, fasta_path, allele_count, repl_samples[i], dir=tempdir)                           
                            synth_image_tensor = create_single_example(
                                params,
                                variant,
                                replicate_bam_path,
                                example_region,
                                repl_samples[i],
                                image_shape=image_shape,
                                realigner=realigner,
                            )
                            repl_encoded_images.append(synth_image_tensor)

                        # Stack all of the image replicates into 4-D tensor (REPLICATES, ROW, COLS, CHANNELS)
                        ac_encoded_images[allele_count] = np.stack(repl_encoded_images)

                # Stack the replicated images for the 3 genotypes (0/0, 0/1, 1/1) into 5-D tensor
                sim_image_tensor = np.stack(ac_encoded_images)
                feature[f"sim/images/encoded"] = _bytes_feature(tf.io.serialize_tensor(sim_image_tensor))

            yield tf.train.Example(features=tf.train.Features(feature=feature))


def _filename_to_compression(filename: str) -> str:
    if filename.endswith(".gz"):
        return "GZIP"
    else:
        return None


def _region_generator(ref_path: str, vcf_path: str, chunk_size=30000000):
    with pysam.FastaFile(ref_path) as ref_file, pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
        for contig, length in zip(ref_file.references, ref_file.lengths):
            if vcf_file.get_tid(contig) != -1:
                for start in range(1, length, chunk_size):
                    yield f"{contig}:{start}-{min(start + chunk_size, length)}"
        

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
):
    def _encoded_example_generator(region=None):
        try:
            # TF seems to encode the region string as bytes
            region = region.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        all_examples = make_vcf_examples(
            params, vcf_path, read_path, sample, image_shape=image_shape, sample_or_label=sample_or_label, simulate=simulate, region=region
        )
        for example in all_examples:
            yield example.SerializeToString()

    def _generate_examples(region):
        return tf.data.Dataset.from_generator(_encoded_example_generator, output_types=(tf.string), args=(region,),)

    with tf.io.TFRecordWriter(output_path, _filename_to_compression(output_path)) as file_writer:
        region_dataset = tf.data.Dataset.from_generator(lambda: _region_generator(params.reference, vcf_path), output_types=(tf.string))
        example_dataset = region_dataset.interleave(
            _generate_examples,
            cycle_length=params.threads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        for example in tqdm(example_dataset, desc="Writing VCF to TFRecords", disable=not progress_bar):
            file_writer.write(example.numpy())


def _features_variant(features):
    return npsv2_pb2.StructuralVariant.FromString(features["variant/encoded"].numpy())


def _example_variant(example):
    encoded_variant = example.features.feature["variant/encoded"].bytes_list.value[0]
    return npsv2_pb2.StructuralVariant.FromString(encoded_variant)


def _example_image(example):
    image_data = tf.io.parse_tensor(example.features.feature["image/encoded"].bytes_list.value[0], tf.uint8).numpy()
    return image_data


def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value[0:3])


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _example_sim_images_shape(example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value[0:5])
    else:
        return (3, 0, None, None, None)


def _example_sim_tensor(example):
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


def example_to_image(example: tf.train.Example, out_path: str, with_simulations=False, margin=10, max_replicates=1):
    # Adapted from DeepVariant
    features = example.features.feature

    image_tensor = _example_image(example)
    shape = image_tensor.shape
    if len(shape) == 3 and shape[2] == 3:
        image_mode = "RGB"
        channels = [0, 1, 2]
    elif len(shape) == 2 or shape[2] == 1:
        image_mode = "L"
        channels = 0
    elif len(shape) == 3 or shape[2] > 3:
        # TODO: Combine all the channels into a single image, perhaps BASE, INSERT_SIZE, ALLELE (with
        # mapq as alpha)...
        image_mode = "RGB"
        channels = [BASE_CHANNEL, ALT_INSERT_SIZE_CHANNEL, ALLELE_CHANNEL]
        #image_mode = "L"
        #channels = ALT_INSERT_SIZE_CHANNEL
    else:
        raise ValueError("Unsupported image shape")

    real_image = Image.fromarray(image_tensor[:, :, channels], mode=image_mode)

    _, replicates, *_ = _example_sim_images_shape(example)
    if with_simulations and replicates > 0:
        height, width, _ = shape
        replicates = min(replicates, max_replicates)
        
        image = Image.new(
            image_mode, (width + 2 * (width + margin), height + replicates * (height + margin))
        )
        image.paste(real_image, (width + margin, 0))

        synth_tensor = _example_sim_tensor(example)
        for ac in (0, 1, 2):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[ac, repl]
                synth_image = Image.fromarray(synth_image_tensor[:, :, channels], mode=image_mode)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def load_example_dataset(filename: str, with_label=False, with_simulations=False) -> tf.data.Dataset:
    # Extract image shape from the first example
    shape, replicates = _extract_metadata_from_first_example(filename)

    proto_features = {
        "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    }
    if with_label:
        proto_features["label"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    if with_simulations and replicates > 0:
        proto_features.update(
            {
                "sim/images/shape": tf.io.FixedLenFeature(shape=[5], dtype=tf.int64),
                "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            }
        )

    def _decode_image(image_feature):
        return tf.reshape(tf.io.decode_raw(image_feature, tf.uint8), shape)

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

    return tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)).map(
        map_func=_process_input
    )

