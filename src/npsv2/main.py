#!/usr/bin/env python3
import argparse, logging


def make_argument_parser():
    parser = argparse.ArgumentParser(
        "npsv2", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        "-v",
        "--verbose",
        help="Verbose logging",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Generate test or training examples
    parser_examples = subparsers.add_parser("examples", help="Generate examples")
    parser_examples.add_argument(
        "-i", "--input", help="Input VCF file.", type=str, dest="vcf", required=True
    )
    parser_examples.add_argument(
        "-b",
        "--bam",
        help="Input alignment (BAM, CRAM) file.",
        type=str,
        dest="bam",
        required=True,
    )
    parser_examples.add_argument(
        "-o", "--output", help="Output file", type=str, required=True
    )

    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.command == "examples":
        from .images import make_vcf_examples
        import tensorflow as tf

        with tf.io.TFRecordWriter(args.output) as dataset:
            all_examples = make_vcf_examples(
                args,
                args.vcf,
                args.bam,
                image_shape=(300, 300),
                sample_or_label="HG002",
            )
            num_examples = 0
            for example in all_examples:
                dataset.write(example.SerializeToString())
                num_examples += 1
            logging.info("Generated %d examples", num_examples)


if __name__ == "__main__":
    main()
