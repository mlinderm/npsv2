import numpy as np
import pysam
import tensorflow as tf
from PIL import Image

from .variant import Variant
from .range import Range

# if ranges.ranges_overlap(variant_utils.variant_range(variant), region)
HEIGHT = 300
MIN_WIDTH = 300
PADDING = 100

CIGAR_ADVANCE_IMAGE = frozenset(
    [
        pysam.CMATCH,
        pysam.CDEL,
        pysam.CREF_SKIP,
        pysam.CSOFT_CLIP,
        pysam.CEQUAL,
        pysam.CDIFF,
    ]
)

CIGAR_BASE_PRESENT = {
    pysam.CMATCH: 255,
    pysam.CDEL: 0,
    pysam.CREF_SKIP: 0,
    pysam.CSOFT_CLIP: 128,
    pysam.CEQUAL: 255,
    pysam.CDIFF: 255,
}


# Adapted from DeepVariant
def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a list of string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))


def _int_feature(list_of_ints):
    """Returns a int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def read_alignment_length(cigar):
    return sum(length for op, length in cigar if op in CIGAR_ADVANCE_IMAGE)


def populate_row_from_cigar(cigar):
    alignment_length = read_alignment_length(cigar)
    base_present = np.zeros(alignment_length, dtype=np.uint8)

    image_pos = 0
    for operation, length in cigar:
        pixel = CIGAR_BASE_PRESENT.get(operation, None)
        if pixel is not None:
            entry_image_end = image_pos + length
            base_present[image_pos:entry_image_end] = pixel
            image_pos = entry_image_end

    return base_present


def create_single_example(
    params, variant, read_path, region, image_shape=None, label=None
):
    # Nucleus regions are 0-indexed half-open
    if isinstance(region, str):
        region = Range.parse_literal(region)
    width = region.length

    image_tensor = np.zeros((HEIGHT, width, 1), dtype=np.uint8)

    count = 0
    with pysam.AlignmentFile(read_path) as alignment_file:
        for read in alignment_file.fetch(
            contig=region.contig, start=region.start, stop=region.end,
        ):
            # TODO: Randomly sample reads when there are more reads than rows
            if count >= HEIGHT:
                break

            cigar = read.cigartuples
            if not cigar:
                continue

            base_present = populate_row_from_cigar(cigar)

            # Determine the mapping between read row and the image block

            # Include "soft clip" in the read
            read_position = read.reference_start
            if cigar[0][0] == pysam.CSOFT_CLIP:
                read_position -= cigar[0][1]

            image_start = read_position - region.start
            image_end = image_start + base_present.size

            # Adjust start and end indices for image
            if image_start < 0:
                base_present = base_present[abs(image_start) :]
                image_start = 0

            if image_end > width:
                base_present = base_present[: width - image_end]
                image_end = width

            image_tensor[count, image_start:image_end, 0] = base_present

            count += 1

    # Create consistent image size
    if image_shape is not None and image_tensor.shape[:2] != image_shape:
        # resize converts to float, directly (however convert_image_dtype assumes floats are in [0-1])
        image_tensor = tf.cast(
            tf.image.resize(image_tensor, image_shape), dtype=tf.uint8
        ).numpy()

    feature = {
        "image/shape": _int_feature(image_tensor.shape),
        "image/encoded": _bytes_feature([image_tensor.tobytes()]),
    }
    if label is not None:
        feature["label"] = _int_feature([label])

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def example_to_image(example, out_path):
    # Adapted from DeepVariant
    features = example.features.feature
    image_data = features["image/encoded"].bytes_list.value[0]
    shape = features["image/shape"].int64_list.value[0:3]
    image_tensor = np.frombuffer(image_data, np.uint8).reshape(shape)

    if len(shape) == 3 and shape[2] == 3:
        image_mode = "RGB"
    elif len(shape) == 2 or shape[2] == 1:
        image_mode = "L"
        image_tensor = image_tensor[:, :, 0]
    else:
        raise ValueError("Unsupported image shape")

    image = Image.fromarray(image_tensor, mode=image_mode)
    image.save(out_path)


def _genotype_to_allele_count(genotype, alleles={1}):
    count = 0
    for gt in genotype:
        if gt == -1:
            return None
        elif gt in alleles:
            count += 1
    return count


def make_vcf_examples(params, vcf_path, read_path, image_shape=None, sample=None):
    with pysam.VariantFile(vcf_path) as vcf_file:
        # Prepare function to extract genotype label
        if sample and isinstance(sample, str):
            sample_index = next(
                (i for i, s in enumerate(vcf_file.header.samples) if s == sample), -1,
            )
            if sample == -1:
                raise ValueError("Sample identifier is not present in the file")
            label_extractor = lambda variant: _genotype_to_allele_count(
                variant.genotype_indices(sample_index)
            )
        else:
            label_extractor = lambda variant: sample

        for record in vcf_file:
            variant = Variant.from_pysam(record)
            assert variant.is_biallelic(), "Multi-allelic sites not yet supported"

            # TODO: Handle odd sized variants when padding
            variant_range = variant.reference_range
            padding = max((MIN_WIDTH - variant_range.length) // 2, PADDING)
            example_region = variant_range.expand(padding)

            label = label_extractor(variant)

            yield create_single_example(
                params,
                variant,
                read_path,
                example_region,
                image_shape=image_shape,
                label=label,
            )


def _example_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value[0:3])


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _extract_shape_from_first_example(filename):
    raw_example = next(iter(tf.data.TFRecordDataset(filenames=filename)))
    example = tf.train.Example.FromString(raw_example.numpy())
    return _example_shape(example)


def load_example_dataset(params, filename, with_labels=False):
    # Extract image shape from the first example
    shape = _extract_shape_from_first_example(filename)

    proto_features = {
        "image/encoded": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    }
    if with_labels:
        proto_features["label"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(
            serialized=proto_string, features=proto_features
        )
        image_tensor = tf.reshape(
            tf.io.decode_raw(parsed_features["image/encoded"], tf.uint8), shape
        )

        if with_labels:
            return image_tensor, parsed_features["label"]
        else:
            return image_tensor, None

    return tf.data.TFRecordDataset(filenames=filename).map(map_func=_process_input)

