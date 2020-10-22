#!/usr/bin/env python3
import argparse, logging, os, subprocess, tempfile
from tqdm import tqdm


def _bwa_index_loaded(reference: str) -> str:
    """Check if bwa index is loaded in shared memory
    
    Args:
        reference (str): Path to reference file
    
    Returns:
        str: Shared reference name if index is loaded into shared memory, None otherwise
    """
    shared_name = os.path.basename(reference)
    indices = subprocess.check_output("bwa shm -l", shell=True, universal_newlines=True, stderr=subprocess.DEVNULL)
    for index in indices.split("\n"):
        if index.startswith(shared_name):
            return shared_name
    return None


def make_argument_parser():
    parser = argparse.ArgumentParser("npsv2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    logging_options = parser.add_mutually_exclusive_group()
    logging_options.add_argument(
        "-d",
        "--debug",
        help="Debug logging",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    logging_options.add_argument(
        "-v", "--verbose", help="Verbose logging", action="store_const", dest="loglevel", const=logging.INFO,
    )

    parser.add_argument(
        "-t", "--tempdir", help="Specify the temp directory", type=str, default=tempfile.gettempdir(),
    )
    parser.add_argument("--threads", help="Number of threads", type=int, default=1)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Generate test or training examples
    parser_examples = subparsers.add_parser("examples", help="Generate train/test examples")

    parser_examples.add_argument("-i", "--input", help="Input VCF file", type=str, dest="vcf", required=True)
    parser_examples.add_argument(
        "-b", "--bam", help="Input alignment (BAM, CRAM) file", type=str, dest="bam", required=True,
    )
    parser_examples.add_argument("-o", "--output", help="Output tfrecords file", type=str, required=True)

    parser_examples.add_argument(
        "-r", "--reference-sequence", help="Reference fasta file", type=str, dest="reference", required=True,
    )

    parser_examples.add_argument("-s", "--sample", help="Sample to use for genotype labels", type=str, default=None)

    parser_examples.add_argument("-n", "--replicates", help="Number of replicates", type=int, default=0)
    parser_examples.add_argument("--flank", help="Flank size for simulation region", type=int, default=1000)

    parser_examples.add_argument("--profile", help="ART profile", type=str, default="HS25")
    parser_examples.add_argument("--read-length", dest="read_length", type=int, help="Read length")
    parser_examples.add_argument("--fragment-mean", dest="fragment_mean", type=float, help="Mean insert size")
    parser_examples.add_argument(
        "--fragment-sd", dest="fragment_sd", type=float, help="Standard deviation of fragment size",
    )
    parser_examples.add_argument("--depth", type=float, help="Mean depth of coverage")

    # Visualize
    visualize_examples = subparsers.add_parser("visualize", help="Convert example to images")
    visualize_examples.add_argument("-i", "--input", help="Input tfrecords file", type=str, required=True)
    visualize_examples.add_argument("-o", "--output", help="Output directory", type=str, required=True)
    visualize_examples.add_argument(
        "-n", "--replicates", help="Max number of replicates to visualize", type=int, default=0
    )
    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.command == "examples":
        from .images import vcf_to_tfrecords

        # Check if shared reference is loaded
        setattr(args, "shared_reference", _bwa_index_loaded(args.reference))
        if args.replicates > 0 and not args.shared_reference:
            logging.warning(
                "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
                args.reference,
            )

        vcf_to_tfrecords(
            args,
            args.vcf,
            args.bam,
            args.output,
            image_shape=(300, 300),
            sample_or_label=args.sample,
            simulate=args.replicates > 0,
            progress_bar=True,
        )

    elif args.command == "visualize":
        from .images import example_to_image
        import tensorflow as tf

        dataset = tf.data.TFRecordDataset(filenames=args.input)
        for i, record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # TODO: Generate variant ID and use that as file name
            image_path = os.path.join(args.output, f"variant{i}.png")
            example_to_image(example, image_path, with_simulations=args.replicates > 0, max_replicates=args.replicates)


if __name__ == "__main__":
    main()
