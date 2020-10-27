import os, subprocess, sys, tempfile
import numpy as np
import pysam
import tensorflow as tf
from PIL import Image
from shlex import quote
from tqdm import tqdm

from .variant import Variant
from .range import Range
from .pileup import Pileup, FragmentTracker
from . import npsv2_pb2

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 300
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

PADDING = 100

BASE_CHANNEL = 0
ALIGNED_BASE_PIXEL = 255
SOFT_CLIP_BASE_PIXEL = 128

REF_INSERT_SIZE_CHANNEL = 1
ALT_INSERT_SIZE_CHANNEL = 2
INSERT_SIZE_MEAN_PIXEL = 128
INSERT_SIZE_SD_PIXEL = 24

def _fragment_zscore(params, fragment_length):
    return (fragment_length - params.fragment_mean) / params.fragment_sd

def create_single_example(params, variant, read_path, region, image_shape=None, label=None):
    if isinstance(region, str):
        region = Range.parse_literal(region)
    width = region.length

    if image_shape and len(image_shape) != 2:
        raise ValueError("Image shape must be (rows, height) sequence")
    elif image_shape:
        tensor_shape = (image_shape[0], width, IMAGE_CHANNELS)
    else:
        tensor_shape = (IMAGE_HEIGHT, width, IMAGE_CHANNELS)

    image_tensor = np.zeros(tensor_shape, dtype=np.uint8)

    pileup = Pileup(region)
    fragments = FragmentTracker()

    with pysam.AlignmentFile(read_path) as alignment_file:
        # Expand query region to capture straddling reads 
        for read in alignment_file.fetch(contig=region.contig, start=region.start-1000, stop=region.end+1000):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            pileup.add_read(read)
            fragments.add_read(read)

    for i in range(width):
        column = pileup[i]

        # TODO: Sample when more reads than rows...
        aligned_pixels = min(column.aligned_bases, tensor_shape[0])
        image_tensor[0:aligned_pixels, i, BASE_CHANNEL] = ALIGNED_BASE_PIXEL
        image_tensor[
            aligned_pixels : min(aligned_pixels + column.soft_clipped_bases, tensor_shape[0]), i, BASE_CHANNEL
        ] = SOFT_CLIP_BASE_PIXEL

    variant_region = variant.reference_region
    left_region = variant.left_flank_region(1000) # TODO: Incorporate CI
    right_region = variant.right_flank_region(1000)

    row = 0
    for fragment in fragments:
        # Since reads were inserted in left-to-fetch, the fragments should be in sorted order
        if row >= IMAGE_HEIGHT:
            break
        elif fragment.fragment_straddles(left_region, right_region, min_aligned=3):
            ref_zscore = _fragment_zscore(params, fragment.fragment_length)
            image_tensor[row, :, REF_INSERT_SIZE_CHANNEL] = INSERT_SIZE_MEAN_PIXEL + ref_zscore * INSERT_SIZE_SD_PIXEL
            alt_zscore = _fragment_zscore(params, fragment.fragment_length + variant.length_change())
            image_tensor[row, :, ALT_INSERT_SIZE_CHANNEL] = INSERT_SIZE_MEAN_PIXEL + alt_zscore * INSERT_SIZE_SD_PIXEL
            row += 1

    # Create consistent image size
    if image_shape and image_tensor.shape[:2] != image_shape:
        # resize converts to float, directly (however convert_image_dtype assumes floats are in [0-1])
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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))


def _int_feature(list_of_ints):
    """Returns a int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _art_read_length(read_length, profile):
    """Make sure read length is compatible ART"""
    if profile in ("HS10", "HS20"):
        return min(read_length, 100)
    elif profile in ("HS25", "HSXn", "HSXt"):
        return min(read_length, 150)
    else:
        return read_length


def _synthesize_variant_data(params, fasta_path, bam_path, allele_count, replicates=1):
    hap_coverage = params.depth / 2
    shared_ref_arg = f"-S {quote(params.shared_reference)}" if params.shared_reference else ""

    synth_commandline = f"synthBAM \
        -t {quote(params.tempdir)} \
        -R {quote(params.reference)} \
        {shared_ref_arg} \
        -c {hap_coverage:0.1f} \
        -m {params.fragment_mean} \
        -s {params.fragment_sd} \
        -l {_art_read_length(params.read_length, params.profile)} \
        -p {params.profile} \
        -i {replicates} \
        -z {allele_count} \
        {fasta_path} \
        {bam_path}"

    synth_result = subprocess.run(synth_commandline, shell=True, stderr=subprocess.PIPE)

    if synth_result.returncode != 0 or not os.path.exists(bam_path):
        print(synth_result.stderr)
        raise RuntimeError(f"Synthesis script failed to generate {bam_path}")


def _replicate_bam_generator(bam_path, replicates, dir=tempfile.gettempdir()):
    if replicates == 1:
        yield bam_path
    else:
        # Split synthetic BAM file into individual replicates
        for i in range(1, replicates + 1):
            read_group = f"synth{i}"
            single_replicate_bam_path = os.path.join(dir, f"{read_group}.bam")

            pysam.view(
                "-b", "-h", "-r", read_group, "-o", single_replicate_bam_path, bam_path, catch_stdout=False,
            )
            pysam.index(single_replicate_bam_path)

            yield single_replicate_bam_path


def make_vcf_examples(
    params, vcf_path: str, read_path: str, image_shape=None, sample_or_label=None, simulate=False, region=None,
):
    with pysam.VariantFile(vcf_path) as vcf_file:
        # Prepare function to extract genotype label
        if sample_or_label and isinstance(sample_or_label, str):
            samples = vcf_file.header.samples
            sample_index = next((i for i, s in enumerate(samples) if s == sample_or_label), -1,)
            if sample_index == -1:
                raise ValueError("Sample identifier is not present in the file")
            vcf_file.subset_samples([sample_or_label])
            label_extractor = lambda variant: _genotype_to_label(variant.genotype_indices(sample_index))
        else:
            vcf_file.subset_samples([])  # Drop all samples
            label_extractor = lambda variant: sample_or_label

        for record in vcf_file.fetch(region=region):
            variant = Variant.from_pysam(record)
            assert variant.is_biallelic(), "Multi-allelic sites not yet supported"

            # TODO: Handle odd sized variants when padding
            variant_region = variant.reference_region
            padding = max((IMAGE_WIDTH - variant_region.length) // 2, PADDING)
            example_region = variant_region.expand(padding)

            # Construct image for "real" data
            image_tensor = create_single_example(params, variant, read_path, example_region, image_shape=image_shape)
            feature = {
                "variant/encoded": _bytes_feature([variant.as_proto().SerializeToString()]),
                "image/shape": _int_feature(image_tensor.shape),
                "image/encoded": _bytes_feature([image_tensor.tobytes()]),
            }

            label = label_extractor(variant)
            if label is not None:
                feature["label"] = _int_feature([label])

            if simulate:
                feature["sim/replicates"] = _int_feature([params.replicates])
                # Generate synthetic training images
                for allele_count in (0, 1, 2):
                    with tempfile.TemporaryDirectory(dir=params.tempdir) as tempdir:
                        # Generate the FASTA file for this zygosity
                        fasta_path, ref_contig, alt_contig = variant.synth_fasta(
                            reference_fasta=params.reference, ac=allele_count, flank=params.flank, dir=tempdir
                        )

                        # Generate synthetic bam files with given number of replicates
                        synthetic_bam_path = os.path.join(tempdir, "replicates.bam")
                        _synthesize_variant_data(
                            params, fasta_path, synthetic_bam_path, allele_count, replicates=params.replicates
                        )

                        synth_encoded_images = []

                        # Split synthetic BAM file into individual replicates
                        for replicate_bam_path in _replicate_bam_generator(
                            synthetic_bam_path, params.replicates, dir=tempdir
                        ):
                            synth_image_tensor = create_single_example(
                                params, variant, replicate_bam_path, example_region, image_shape=image_shape
                            )
                            synth_encoded_images.append(synth_image_tensor.tobytes())

                        feature[f"sim/{allele_count}/images/encoded"] = _bytes_feature(synth_encoded_images)

            yield tf.train.Example(features=tf.train.Features(feature=feature))


def _filename_to_compression(filename: str) -> str:
    if filename.endswith(".gz"):
        return "GZIP"
    else:
        return None


def _region_generator(vcf_path: str, chunk_size=30000000):
    # TODO: Chunk within chromosomes
    with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
        tid = 0
        while vcf_file.is_valid_tid(tid):
            yield vcf_file.get_reference_name(tid)
            tid += 1


def vcf_to_tfrecords(
    params,
    vcf_path: str,
    read_path: str,
    output_path: str,
    image_shape=None,
    sample_or_label=None,
    simulate=False,
    progress_bar=False,
):
    def _encoded_example_generator(region=None):
        all_examples = make_vcf_examples(
            params, vcf_path, read_path, image_shape, sample_or_label, simulate=simulate, region=region
        )
        for example in all_examples:
            yield example.SerializeToString()

    def _generate_examples(region):
        return tf.data.Dataset.from_generator(_encoded_example_generator, output_types=(tf.string), args=(region,),)

    with tf.io.TFRecordWriter(output_path, _filename_to_compression(output_path)) as file_writer:
        region_dataset = tf.data.Dataset.from_generator(lambda: _region_generator(vcf_path), output_types=(tf.string))
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


def _extract_image(example, shape):
    image_data = example.features.feature["image/encoded"].bytes_list.value[0]
    return np.frombuffer(image_data, np.uint8).reshape(shape)


def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value[0:3])


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _example_sim_replicates(example):
    if "sim/replicates" in example.features.feature:
        return int(example.features.feature["sim/replicates"].int64_list.value[0])
    else:
        return 0


def _extract_metadata_from_first_example(filename):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())
    return _example_image_shape(example), _example_sim_replicates(example)


def example_to_image(example: tf.train.Example, out_path: str, with_simulations=False, margin=10, max_replicates=1):
    # Adapted from DeepVariant
    features = example.features.feature

    shape = features["image/shape"].int64_list.value[0:3]
    if len(shape) == 3 and shape[2] == 3:
        image_mode = "RGB"
        channels = [0, 1, 2]
    elif len(shape) == 2 or shape[2] == 1:
        image_mode = "L"
        channels = 0
    else:
        raise ValueError("Unsupported image shape")

    image_tensor = _extract_image(example, shape)
    real_image = Image.fromarray(image_tensor[:, :, channels], mode=image_mode)

    if with_simulations and "sim/replicates" in features:
        width, IMAGE_HEIGHT, _ = shape
        replicates = min(features["sim/replicates"].int64_list.value[0], max_replicates)

        image = Image.new(
            image_mode, (width + 2 * (width + margin), IMAGE_HEIGHT + replicates * (IMAGE_HEIGHT + margin))
        )
        image.paste(real_image, (width + margin, 0))

        for ac in (0, 1, 2):
            synth_data = features[f"sim/{ac}/images/encoded"].bytes_list.value
            for repl in range(min(replicates, max_replicates)):
                synth_tensor = np.frombuffer(synth_data[repl], np.uint8).reshape(shape)
                synth_image = Image.fromarray(synth_tensor[:, :, channels], mode=image_mode)

                coord = (ac * (width + margin), (repl + 1) * (IMAGE_HEIGHT + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def load_example_dataset(filename: str, with_label=False, with_simulations=False) -> tf.data.Dataset:
    # Extract image shape from the first example
    shape, replicates = _extract_metadata_from_first_example(filename)

    proto_features = {
        "variant/encoded": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    }
    if with_label:
        proto_features["label"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    if with_simulations and replicates > 0:
        proto_features.update(
            {
                "sim/replicates": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
                "sim/0/images/encoded": tf.io.FixedLenFeature(shape=[replicates], dtype=tf.string),
                "sim/1/images/encoded": tf.io.FixedLenFeature(shape=[replicates], dtype=tf.string),
                "sim/2/images/encoded": tf.io.FixedLenFeature(shape=[replicates], dtype=tf.string),
            }
        )

    def _decode_image(image_feature):
        return tf.reshape(tf.io.decode_raw(image_feature, tf.uint8), shape)

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            "variant/encoded": parsed_features["variant/encoded"][0],
            "image": _decode_image(parsed_features["image/encoded"]),
        }
        if with_simulations:
            for ac in (0, 1, 2):
                # dytpe is deprecated in tf 2.3 (but only 2.2 is available through conda)
                features[f"sim/{ac}/images"] = tf.map_fn(
                    _decode_image, parsed_features[f"sim/{ac}/images/encoded"], dtype=tf.uint8
                )

        if with_label:
            return features, parsed_features["label"]
        else:
            return features, None

    return tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)).map(
        map_func=_process_input
    )

