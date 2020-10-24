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

def _configure_gpu():
    """
    Configure GPU options (seems to be required for RTX GPUs)
    """
    import tensorflow as tf
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:    
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass


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
    parser_visualize = subparsers.add_parser("visualize", help="Convert example to images")
    parser_visualize.add_argument("-i", "--input", help="Input tfrecords file", type=str, required=True)
    parser_visualize.add_argument("-o", "--output", help="Output directory", type=str, required=True)
    parser_visualize.add_argument(
        "-n", "--replicates", help="Max number of replicates to visualize", type=int, default=0
    )

    # Training
    parser_train = subparsers.add_parser("train", help="Convert example to images")
    parser_train.add_argument("-i", "--input", help="Input tfrecords file", type=str, required=True)
    parser_train.add_argument("-o", "--output", help="Path to save model", type=str, required=True)

    # Evaluation
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate model")
    parser_evaluate.add_argument("-m", "--model", help="Saved model", type=str, required=True)
    parser_evaluate.add_argument("-d", "--dataset", help="Input tfrecords file", type=str, required=True)
    
    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    _configure_gpu()

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
        import tensorflow as tf
        from .images import example_to_image

        dataset = tf.data.TFRecordDataset(filenames=args.input)
        for i, record in enumerate(tqdm(dataset, desc="Generating images for each variant")):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # TODO: Generate variant ID and use that as file name
            image_path = os.path.join(args.output, f"variant{i}.png")
            example_to_image(example, image_path, with_simulations=args.replicates > 0, max_replicates=args.replicates)

    elif args.command == "train":
        import tensorflow as tf
        from .training import train
        
        train(args.input, args.output)

    elif args.command == "evaluate":
        import tensorflow as tf
        from .training import evaluate_model

        genotype_concordance, nonreference_concordance, confusion_matrix = evaluate_model(args.model, args.dataset)
        print(f"Accuracy -  Genotype concordance: {genotype_concordance:.3}, Non-reference Concordance: {nonreference_concordance:.3}")

        # Print the confusion matrix
        tf.print("Confusion Matrix")
        tf.print("\t\t\tTest\t")
        tf.print("T", "", "0/0", "0/1", "1/1", sep="\t")
        tf.print("R", "0/0", *confusion_matrix[0,:], sep="\t", summarize=-1)
        tf.print("U", "0/1", *confusion_matrix[1,:], sep="\t", summarize=-1)
        tf.print("E", "1/1", *confusion_matrix[2,:], sep="\t", summarize=-1)

if __name__ == "__main__":
    main()
