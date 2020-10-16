import numpy as np
from nucleus.util import cigar, ranges, variant_utils
from nucleus.io import sam, tfrecord, vcf
from nucleus.protos import cigar_pb2
import tensorflow as tf
from PIL import Image

# if ranges.ranges_overlap(variant_utils.variant_range(variant), region)
HEIGHT=300
MIN_WIDTH=300
PADDING=100

CIGAR_ADVANCE_IMAGE = frozenset([
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH,
    cigar_pb2.CigarUnit.DELETE,
    cigar_pb2.CigarUnit.SKIP,
    cigar_pb2.CigarUnit.CLIP_SOFT,
    cigar_pb2.CigarUnit.SEQUENCE_MATCH,
    cigar_pb2.CigarUnit.SEQUENCE_MISMATCH
])

CIGAR_BASE_PRESENT = {
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH: 255,
    cigar_pb2.CigarUnit.DELETE: 0,
    cigar_pb2.CigarUnit.SKIP: 0,
    cigar_pb2.CigarUnit.CLIP_SOFT: 128,
    cigar_pb2.CigarUnit.SEQUENCE_MATCH: 255,
    cigar_pb2.CigarUnit.SEQUENCE_MISMATCH: 255,
}


# Adapted from DeepVariant
def _bytes_feature(list_of_strings):
  """Returns a bytes_list from a list of string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))

def _int_feature(list_of_ints):
  """Returns a int64_list from a list of int / bool."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=list_of_ints))

def read_alignment_length(cigar):
    return sum(entry.operation_length
             for entry in cigar
             if entry.operation in CIGAR_ADVANCE_IMAGE)


def populate_row_from_cigar(cigar):
    alignment_length = read_alignment_length(cigar)
    base_present = np.zeros(alignment_length, dtype=np.uint8)
    
    image_pos = 0
    for cigar_entry in cigar:
        pixel = CIGAR_BASE_PRESENT.get(cigar_entry.operation, None)
        if pixel is not None:
            entry_image_end = image_pos+cigar_entry.operation_length
            base_present[image_pos:entry_image_end] = pixel
            image_pos = entry_image_end

    return base_present


def create_single_example(params, variant, read_path, region_or_string, image_shape=None, label=None):
    # Nucleus regions are 0-indexed half-open
    if isinstance(region_or_string, str):
        region_or_string = ranges.parse_literal(region_or_string)
    width = ranges.length(region_or_string)
    
    image_tensor = np.zeros((HEIGHT, width, 1), dtype=np.uint8)

    count = 0
    with sam.SamReader(read_path) as read_reader:
        reads = read_reader.query(region_or_string)
        for read in reads:            
            # TODO: Randomly sample reads when there are more reads than rows
            if count >= HEIGHT:
                break

            cigar = read.alignment.cigar
            if not cigar:
                continue
            
            base_present = populate_row_from_cigar(cigar)
          
            # Determine the mapping between read row and the image block
            
            # Include "soft clip" in the read
            read_position = read.alignment.position.position
            if cigar[0].operation == cigar_pb2.CigarUnit.CLIP_SOFT:
                read_position -= cigar[0].operation_length

            image_start = read_position - region_or_string.start
            image_end = image_start + base_present.size

            # Adjust start and end indices for image
            if image_start < 0:
                base_present = base_present[abs(image_start):]
                image_start = 0
            
            if image_end > width:
                base_present = base_present[:width-image_end]
                image_end = width
      
            image_tensor[count, image_start:image_end, 0] = base_present
            
            count += 1
    
    # Create consistent image size
    if image_shape is not None and image_tensor.shape[:2] != image_shape:
        # resize converts to float, directly (however convert_image_dtype assumes floats are in [0-1])
        image_tensor = tf.cast(tf.image.resize(image_tensor, image_shape), dtype=tf.uint8).numpy()

    feature = {
        "image/shape": _int_feature(image_tensor.shape),
        "image/encoded": _bytes_feature([image_tensor.tobytes()]),
    }
    if label is not None:
        feature["label"] =  _int_feature([label])

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def example_to_image(example, out_path):
    # Adapted from DeepVariant
    features = example.features.feature
    image_data = features['image/encoded'].bytes_list.value[0]
    shape = features['image/shape'].int64_list.value[0:3]
    image_tensor = np.frombuffer(image_data, np.uint8).reshape(shape)

    if len(shape) == 3 and shape[2] == 3:
        image_mode = 'RGB'
    elif len(shape) == 2 or shape[2] == 1:
        image_mode = 'L'
        image_tensor = image_tensor[:,:,0]
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


def make_vcf_examples(params, vcf_path, read_path, image_shape=None, sample_or_label=None):
    with vcf.VcfReader(input_path=vcf_path) as vcf_reader:
        # Prepare function to extract genotype label
        if sample_or_label is not None and isinstance(sample_or_label, str):
            sample_index = next((i for i, s in enumerate(vcf_reader.header.sample_names) if s == sample_or_label), -1)
            if sample_or_label == -1:
                raise ValueError("Sample identifier is not present in the file")
            label_extractor = lambda variant: _genotype_to_allele_count(variant.calls[sample_index].genotype)
        else:
            label_extractor = lambda variant: sample_or_label
        
        
        for variant in vcf_reader:
            assert variant_utils.is_biallelic(variant), "Multi-allelic sites not yet supported"
             
            # TODO: Handle odd sized variants when padding
            variant_range = variant_utils.variant_range(variant)   
            padding = max((MIN_WIDTH - ranges.length(variant_range)) // 2, PADDING)
            example_region = ranges.expand(variant_range, padding)

            label = label_extractor(variant)

            yield create_single_example(params, variant, read_path, example_region, image_shape=image_shape, label=label)

def _example_shape(example):
    return tuple(example.features.feature['image/shape'].int64_list.value[0:3])

def _example_label(example):
    return int(example.features.feature['label'].int64_list.value[0])

def _extract_shape_from_first_example(filename):
    example = next(tfrecord.read_tfrecords(filename, max_records=1))
    return _example_shape(example)



def load_example_dataset(params, filename, with_labels=False):
    # Extract image shape from the first example
    shape = _extract_shape_from_first_example(filename)
    
    proto_features = {
        'image/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'image/shape': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    }
    if with_labels:
        proto_features["label"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)
        image_tensor = tf.reshape(tf.io.decode_raw(parsed_features["image/encoded"], tf.uint8), shape)
        
        if with_labels:
            return image_tensor, parsed_features["label"]
        else:
            return image_tensor, None
    
    return tf.data.TFRecordDataset(filenames=filename).map(map_func=_process_input)

