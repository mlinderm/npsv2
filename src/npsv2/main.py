#!/usr/bin/env python3
import argparse, logging, os, subprocess, sys, tempfile
from omegaconf import DictConfig, OmegaConf
import hydra
import tensorflow as tf
from tqdm import tqdm


def _configure_gpu():
    """
    Configure GPU options (seems to be required for RTX GPUs)
    """
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth error:", e, file=sys.stderr)
        pass


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


def _check_shared_reference(cfg: DictConfig):
    cfg.shared_reference = _bwa_index_loaded(hydra.utils.to_absolute_path(cfg.reference))
    if cfg.simulation.replicates > 0 and not cfg.shared_reference:
        logging.warning(
            "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
            cfg.reference,
        )

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.command == "images":
        import ray
        from .images import vcf_to_tfrecords
        from .sample import Sample, sample_name_from_bam

        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        vcf_to_tfrecords(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            hydra.utils.to_absolute_path(cfg.reads),
            hydra.utils.to_absolute_path(cfg.output),
            sample,
            sample_or_label=cfg.sample,
            simulate=cfg.simulation.replicates > 0,
            progress_bar=True,
        )

    elif cfg.command == "visualize":
        from .images import example_to_image, _filename_to_compression

        dataset = tf.data.TFRecordDataset(filenames=hydra.utils.to_absolute_path(cfg.input), compression_type=_filename_to_compression(hydra.utils.to_absolute_path(cfg.input)))
        for i, record in enumerate(tqdm(dataset, desc="Generating images for each variant")):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # TODO: Generate variant ID and use that as file name
            # TODO: Manage the Hydra generated working directory
            image_path = os.path.join(os.getcwd(), f"variant{i}.png")
            example_to_image(cfg, example, image_path, with_simulations=True, max_replicates=cfg.simulation.replicates)

    elif cfg.command == "train":
        from .images import _extract_metadata_from_first_example, load_example_dataset
        
        _configure_gpu()

        tfrecords_paths = [cfg.input] if isinstance(cfg.input, str) else cfg.input
        tfrecords_paths = [hydra.utils.to_absolute_path(p) for p in tfrecords_paths]
        
        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
        model = hydra.utils.instantiate(cfg.model, image_shape, replicates)

        dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True)
        model.fit(cfg, dataset)
    
        model_path = os.path.join(os.getcwd(), "model.h5")
        logging.info("Saving model in: %s", model_path)
        model.save(model_path)

    elif cfg.command == "evaluate":
        import numpy as np
        import pandas as pd
        from .images import _extract_metadata_from_first_example, load_example_dataset
        from . import npsv2_pb2

        _configure_gpu()

        tfrecords_paths = [cfg.input] if isinstance(cfg.input, str) else cfg.input
        tfrecords_paths = [hydra.utils.to_absolute_path(p) for p in tfrecords_paths]

        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
        cfg.model.model_path = hydra.utils.to_absolute_path(cfg.model.model_path)
        model = hydra.utils.instantiate(cfg.model, image_shape, 1)

        rows = []
        for features, original_label in load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True):
            # Extract metadata for the variant
            variant_proto = npsv2_pb2.StructuralVariant.FromString(features.pop("variant/encoded").numpy())
            
            # Predict genotype
            dataset = tf.data.Dataset.from_tensors((features, original_label))
            genotypes, distances, *_  = model.predict(cfg, dataset)

            # Construct the DataFrame rows
            rows.append(pd.DataFrame({
                "SVLEN": variant_proto.svlen,
                "LABEL": original_label,
                "AC": tf.math.argmax(genotypes, axis=1),
            }))

        table = pd.concat(rows, ignore_index=True)
        table["AC"] = pd.Categorical(table["AC"], categories=[0, 1, 2])
        table["LABEL"] = pd.Categorical(table["LABEL"], categories=[0, 1, 2])
        table["MATCH"] = table.LABEL == table.AC

        # Print various metrics
        gt_conc = np.mean(table.MATCH)
        nr_conc = np.mean((table.LABEL != 0) == (table.AC != 0))
        print(f"Accuracy - Genotype concordance: {gt_conc:.3}, Non-reference Concordance: {nr_conc:.3}")

        confusion_matrix = pd.crosstab(table.LABEL, table.AC, rownames=["Truth"], colnames=["Test"], margins=True, dropna=False)
        print(confusion_matrix)

        svlen_bins = pd.cut(np.abs(table.SVLEN), [50, 100, 300, 1000, np.iinfo(np.int32).max], right=False)
        print(table.groupby(svlen_bins)["MATCH"].mean())

    elif cfg.command == "genotype":
        from .sample import Sample, sample_name_from_bam
        from .genotyping import genotype_vcf

        _check_shared_reference(cfg)

        stats_paths = [cfg.stats_path] if isinstance(cfg.stats_path, str) else cfg.stats_path
        reads_paths = [cfg.reads] if isinstance(cfg.reads, str) else cfg.reads

        samples = {}
        for stat_path in stats_paths:
            sample = Sample.from_json(hydra.utils.to_absolute_path(stat_path))
            samples[sample.name] = sample
        for reads_path in reads_paths:
            reads_path = hydra.utils.to_absolute_path(reads_path)
            samples[sample_name_from_bam(reads_path)].bam = reads_path
       
        # Make sure model path is absolute
        cfg.model.model_path = hydra.utils.to_absolute_path(cfg.model.model_path)

        genotype_vcf(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            samples,
            hydra.utils.to_absolute_path(cfg.output),
            progress_bar=True,
        )

if __name__ == "__main__":
    main()

# def _image_size(arg: str):
#     try:
#         rows, cols = map(int, arg.split(","))
#         return (rows, cols)
#     except:
#         raise argparse.ArgumentTypeError("Image size must be of the form 'row,col'")

# def simulation_options(parser):
#     parser.add_argument(
#         "--sample-ref", help="Sample reference genotypes instead of simulating", action="store_true", default=False
#     )
#     parser.add_argument(
#         "--exclude-bed",
#         type=str,
#         help="Tabix-indexed BED file of regions to exclude from random sampling",
#         default=None,
#     )
#     parser.add_argument(
#         "--augment", help="Augment sequencing parameters when generating examples", action="store_true", default=False
#     )
#     parser.add_argument(
#         "--windowed",
#         dest="single_image",
#         default=True,
#         action="store_false",
#         help="Generate windowed pileup images instead of (compressed) single image"
#     )

# def make_argument_parser():
#     parser = argparse.ArgumentParser("npsv2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     logging_options = parser.add_mutually_exclusive_group()
#     logging_options.add_argument(
#         "-d",
#         "--debug",
#         help="Debug logging",
#         action="store_const",
#         dest="loglevel",
#         const=logging.DEBUG,
#         default=logging.WARNING,
#     )
#     logging_options.add_argument(
#         "-v", "--verbose", help="Verbose logging", action="store_const", dest="loglevel", const=logging.INFO,
#     )

#     parser.add_argument(
#         "-t", "--tempdir", help="Specify the temp directory", type=str, default=tempfile.gettempdir(),
#     )
#     parser.add_argument("--threads", help="Number of threads", type=int, default=1)

#     subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

#     # Generate test or training examples
#     parser_examples = subparsers.add_parser(
#         "examples", help="Generate train/test examples", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )

#     parser_examples.add_argument("-i", "--input", help="Input VCF file", type=str, dest="vcf", required=True)
#     parser_examples.add_argument(
#         "-b", "--bam", help="Input alignment (BAM, CRAM) file", type=str, dest="bam", required=True,
#     )
#     parser_examples.add_argument("-o", "--output", help="Output tfrecords file", type=str, required=True)

#     parser_examples.add_argument(
#         "-r", "--reference-sequence", help="Reference fasta file", type=str, dest="reference", required=True,
#     )

#     parser_examples.add_argument("--size", help="Size of images to generate", type=_image_size, default=(100, 300))
#     parser_examples.add_argument("-s", "--sample", help="Sample to use for genotype labels", type=str, default=None)

#     parser_examples.add_argument("-n", "--replicates", help="Number of replicates", type=int, default=0)
#     parser_examples.add_argument("--flank", help="Flank size for simulation region", type=int, default=1000)

#     sample_stats_group = parser_examples.add_argument_group()
#     sample_stats_group.add_argument("--profile", type=str, help="ART profile", default="HS25")
#     sample_stats_group.add_argument("--read-length", dest="read_length", type=int, help="Read length", default=150)
#     sample_stats_group.add_argument(
#         "--fragment-mean", dest="fragment_mean", type=float, help="Mean insert size", default=500
#     )
#     sample_stats_group.add_argument(
#         "--fragment-sd", dest="fragment_sd", type=float, help="Standard deviation of fragment size", default=150,
#     )
#     sample_stats_group.add_argument("--depth", type=float, help="Mean depth of coverage", default=30)

#     parser_examples.add_argument(
#         "--stats-path", type=str, help="Path to stats JSON file generated by preprocessing command",
#     )
#     simulation_options(parser_examples)

#     # Visualize
#     parser_visualize = subparsers.add_parser(
#         "visualize", help="Convert example to images", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_visualize.add_argument("-i", "--input", help="Input tfrecords file", type=str, required=True)
#     parser_visualize.add_argument("-o", "--output", help="Output directory", type=str, required=True)
#     parser_visualize.add_argument(
#         "-n", "--replicates", help="Max number of replicates to visualize", type=int, default=0
#     )

#     # Training
#     parser_train = subparsers.add_parser(
#         "train", help="Convert example to images", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_train.add_argument("-i", "--input", help="Input tfrecords file", type=str, action="append", required=True)
#     parser_train.add_argument("-o", "--output", help="Path to save model", type=str, required=True)
#     parser_train.add_argument("--epochs", help="Maximum number of epochs", type=int, default=5)
#     parser_train.add_argument("--lr", dest="learning_rate", help="(Initial) learning rate", type=float, default=0.004)
#     parser_train.add_argument(
#         "--log-dir", dest="log_dir", help="Directory to write time-stamped TensorBoard logs", type=str, default=None
#     )
#     parser_train.add_argument("-m", "--model", help="Saved model to use as a starting point for training", type=str, default=None)

#     # Evaluation
#     parser_evaluate = subparsers.add_parser(
#         "evaluate", help="Evaluate model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_evaluate.add_argument(
#         "-i", "--input", help="Input tfrecords file", type=str, action="append", required=True
#     )
#     parser_evaluate.add_argument("-m", "--model", help="Saved model", type=str, required=True)

#     # Plot embeddings
#     parser_embeddings = subparsers.add_parser(
#         "embeddings", help="Plot embeddings", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_embeddings.add_argument(
#         "-i", "--input", help="Input tfrecords file", type=str, action="append", required=True
#     )
#     parser_embeddings.add_argument("-m", "--model", help="Saved model", type=str, required=True)

#     # Simulate effect of GC on sequencing coverage
#     parser_sim_gc = subparsers.add_parser("simgc", help="Model effect of GC on sequencing coverage")
#     parser_sim_gc.add_argument(
#         "--fasta-path",
#         dest="fasta_path",
#         type=str,
#         help="Path to FASTA file used to generate the synthetic reads",
#         required=True,
#     )
#     parser_sim_gc.add_argument(
#         "--stats-path", dest="stats_path", type=str, help="Path to samples stats JSON file", required=True,
#     )
#     parser_sim_gc.add_argument("-i", "--input", help="Input SAM file.", type=str, dest="input", required=True)
#     parser_sim_gc.add_argument(
#         "--max-norm-covg", dest="max_norm_covg", help="Max normalized coverage", type=float, default=2.0,
#     )

#     # Genotyping
#     parser_genotype = subparsers.add_parser(
#         "genotype", help="Genotype VCF", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_genotype.add_argument("-m", "--model", help="Saved model", type=str, required=True)
#     parser_genotype.add_argument("-i", "--input", help="Input VCF file", type=str, required=True)
#     parser_genotype.add_argument(
#         "-b", "--bam", help="Input alignment (BAM, CRAM) file", type=str, dest="bams", action="append", required=True,
#     )
#     parser_genotype.add_argument("-o", "--output", help="Output VCF file", type=str, required=True)
#     parser_genotype.add_argument("-n", "--replicates", help="Number of replicates", type=int, default=1)
#     parser_genotype.add_argument(
#         "-r", "--reference-sequence", help="Reference fasta file", type=str, dest="reference", required=True,
#     )
#     parser_genotype.add_argument("--flank", help="Flank size for simulation region", type=int, default=1000)
#     parser_genotype.add_argument(
#         "--stats-path", help="Path to stats JSON file generated by preprocessing command", type=str, dest="stats_paths", action="append"
#     )
#     parser_genotype.add_argument("--size", help="Size of images to generate", type=_image_size, default=(100, 300))
#     simulation_options(parser_genotype)
    
#     # Refine VCF
#     parser_refine = subparsers.add_parser(
#         "refine", help="Refine proposal VCF", formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser_refine.add_argument("-i", "--input", help="Input VCF file", type=str, required=True)
#     parser_refine.add_argument("-o", "--output", help="Output VCF file", type=str, required=True)
    
#     return parser





# def main():
#     parser = make_argument_parser()
#     args = parser.parse_args()

#     _configure_gpu()

#     if args.command == "examples":
#         import ray
#         from .images import vcf_to_tfrecords
#         from .sample import Sample, sample_name_from_bam

#         _check_shared_reference(args)

#         if args.stats_path:
#             sample = Sample.from_json(args.stats_path)
#         else:
#             sample = Sample(
#                 sample_name_from_bam(args.bam),
#                 mean_coverage=args.depth,
#                 read_length=args.read_length,
#                 sequencer=args.profile,
#                 mean_insert_size=args.fragment_mean,
#                 std_insert_size=args.fragment_sd,
#             )

#         vcf_to_tfrecords(
#             args,
#             args.vcf,
#             args.bam,
#             args.output,
#             sample,
#             image_shape=args.size,
#             sample_or_label=args.sample,
#             simulate=args.replicates > 0,
#             progress_bar=True,
#         )

#     elif args.command == "visualize":
#         from .images import example_to_image, _filename_to_compression

#         dataset = tf.data.TFRecordDataset(filenames=args.input, compression_type=_filename_to_compression(args.input))
        
#         # Filter by region if specified
        
#         for i, record in enumerate(tqdm(dataset, desc="Generating images for each variant")):
#             example = tf.train.Example()
#             example.ParseFromString(record.numpy())

#             # TODO: Generate variant ID and use that as file name
#             image_path = os.path.join(args.output, f"variant{i}.png")
#             example_to_image(example, image_path, with_simulations=args.replicates > 0, max_replicates=args.replicates)

#     elif args.command == "train":
#         from .training import train

#         train(args, args.input, args.output, starting_model_path=args.model)

#     elif args.command == "evaluate":
#         import numpy as np
#         import pandas as pd
#         from .training import evaluate_model

#         table = evaluate_model(args, args.input, args.model)

#         # Print various metrics
#         gt_conc = np.mean(table.MATCH)
#         nr_conc = np.mean((table.LABEL != 0) == (table.AC != 0))
#         print(f"Accuracy - Genotype concordance: {gt_conc:.3}, Non-reference Concordance: {nr_conc:.3}")

#         confusion_matrix = pd.crosstab(table.LABEL, table.AC, rownames=["Truth"], colnames=["Test"], margins=True, dropna=False)
#         print(confusion_matrix)

#         svlen_bins = pd.cut(np.abs(table.SVLEN), [50, 100, 300, 1000, np.iinfo(np.int32).max], right=False)
#         print(table.groupby(svlen_bins)["MATCH"].mean())

#     elif args.command == "embeddings":
#         from .training import visualize_embeddings

#         visualize_embeddings(args.input, args.model)

#     elif args.command == "simgc":
#         from .simulation import filter_reads_gc

#         filter_reads_gc(
#             args.stats_path, args.fasta_path, args.input, "/dev/stdout", max_norm_covg=args.max_norm_covg,
#         )

#     elif args.command == "genotype":
#         from .sample import Sample, sample_name_from_bam
#         from .genotyping import genotype_vcf

#         samples = {}
#         if args.stats_paths:
#             for stat_path in args.stats_paths:
#                 sample = Sample.from_json(stat_path)
#                 samples[sample.name] = sample

#             for bam in args.bams:
#                 samples[sample_name_from_bam(bam)].bam = bam
#         else:
#             for bam in args.bams:
#                 sample = Sample(
#                     sample_name_from_bam(bam),
#                     bam=bam,
#                     mean_coverage=args.depth,
#                     read_length=args.read_length,
#                     sequencer=args.profile,
#                     mean_insert_size=args.fragment_mean,
#                     std_insert_size=args.fragment_sd,
#                 )
#                 samples[sample.name] = sample

#         _check_shared_reference(args)

#         genotype_vcf(args, args.model, args.input, samples, args.output, args.size, progress_bar=True)

#     elif args.command == "refine":
#         from .propose.refine import refine_vcf

#         refine_vcf(args, args.input, args.output, progress_bar=True)

# if __name__ == "__main__":
#     main()
