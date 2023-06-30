#!/usr/bin/env python3
import argparse, json, logging, os, random, re, subprocess, shutil, sys, tempfile, typing
from omegaconf import ListConfig, DictConfig, OmegaConf
import hydra
import tensorflow as tf
from tqdm import tqdm
from .simulation import bwa_index_loaded

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


def _check_shared_reference(cfg: DictConfig):
    """Check if BWA shared index is loaded, loading it if specified configuration"""
    if cfg.simulation.replicates > 0:
        cfg.shared_reference = bwa_index_loaded(hydra.utils.to_absolute_path(cfg.reference), load=cfg.load_reference)
        if not cfg.shared_reference:
            logging.warning(
                "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
                cfg.reference,
            )


def _is_tfrecords_file(filename: str) -> bool:
    """Return true if tfrecords file (has '.tfrecords[.gz]' extension)"""
    return filename.endswith((".tfrecords", "tfrecords.gz"))


def _as_list(item_or_list):
    """Convert scalar argument to list, or pass list through"""
    return item_or_list if isinstance(item_or_list, (list, ListConfig)) else [item_or_list]


def _make_paths_absolute(cfg: DictConfig, keys: typing.Iterable[str]):
    """Make list of hydra configuration keys, e.g. 'pileup.snv_vcf_input' absolute paths"""
    for key in keys:
        if not OmegaConf.is_missing(cfg, key) and OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, hydra.utils.to_absolute_path(OmegaConf.select(cfg, key)))


def _get_file(cfg: DictConfig, path_or_url: str) -> str:
    """Download file from URL if not found locally or already cached"""
    if os.path.exists(path_or_url):
        return hydra.utils.to_absolute_path(path_or_url)
    else:
        # Attempt to use cached copy or download from URL
        return tf.keras.utils.get_file(origin=path_or_url, cache_subdir="models", cache_dir=cfg.cache_dir)

def _normalize_fname(str_or_bytes: typing.Union[bytes, str]) -> str:
    """Normalize string or bytes to string"""
    if isinstance(str_or_bytes, bytes):
        return str_or_bytes.decode("utf-8")
    else:
        return str_or_bytes

# Resolvers for use with Hydra
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("swap_ext", lambda path, old_ext, new_ext: re.sub(old_ext + "$", new_ext, path))
OmegaConf.register_new_resolver("strip_ext", lambda path: os.path.splitext(path)[0])
OmegaConf.register_new_resolver("escape", lambda path: str(path).replace("[","").replace("]","").replace(", ","_")) 

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.command == "images":
        from .images import vcf_to_tfrecords
        from .sample import Sample, sample_name_from_bam

        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "images.tfrecords.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        vcf_to_tfrecords(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            hydra.utils.to_absolute_path(cfg.reads),
            output,
            sample,
            sample_or_label=cfg.sample,
            simulate=cfg.simulation.replicates > 0,
            progress_bar=True,
        )

    elif cfg.command == "visualize":
        from .images import example_to_image, _filename_to_compression, make_vcf_examples 
        from .sample import Sample, sample_name_from_bam
        # TODO: Generate variant ID and use that as file name
        # TODO: Manage the Hydra generated working directory

        input_path = hydra.utils.to_absolute_path(cfg.input)
        if _is_tfrecords_file(input_path):
            dataset = tf.data.TFRecordDataset(filenames=input_path, compression_type=_filename_to_compression(input_path))
            for i, record in enumerate(tqdm(dataset, desc="Generating images for each variant")):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())

                image_path = os.path.join(os.getcwd(), f"variant{i}.png")
                example_to_image(cfg, example, image_path, with_simulations=True, max_replicates=cfg.simulation.replicates, render_channels=cfg.visualize.render_channels)
        else:  # Assume it is a VCF file
            _check_shared_reference(cfg)
            sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

            examples = make_vcf_examples(cfg, input_path, hydra.utils.to_absolute_path(cfg.reads), sample, simulate=True)
            for i, example in enumerate(tqdm(examples, desc="Generating images for each variant")):
                image_path = os.path.join(os.getcwd(), f"variant{i}.png")
                example_to_image(cfg, example, image_path, with_simulations=True, max_replicates=cfg.simulation.replicates, render_channels=cfg.visualize.render_channels)

    elif cfg.command == "train":
        from .images import _extract_metadata_from_first_example, load_example_dataset
        
        _configure_gpu()

        # Make sure paths are absolute
        tfrecords_paths = [cfg.input] if isinstance(cfg.input, str) else cfg.input
        tfrecords_paths = [hydra.utils.to_absolute_path(p) for p in tfrecords_paths]

        if cfg.training.validation_input and isinstance(cfg.training.validation_input, str):
            validation_tfrecords_paths = [cfg.training.validation_input]
        elif cfg.training.validation_input:
            validation_tfrecords_paths = cfg.training.validation_input
        else:
            validation_tfrecords_paths = []
        validation_tfrecords_paths = [hydra.utils.to_absolute_path(p) for p in validation_tfrecords_paths]

        if cfg.training.validation_split and len(validation_tfrecords_paths) == 0 and len(tfrecords_paths) > 1:
            # Construct validation datasets from input files
            random.shuffle(tfrecords_paths)
            logging.info("Using random selection of %d input files for validation", cfg.training.validation_split)
            validation_tfrecords_paths, tfrecords_paths = tfrecords_paths[:cfg.training.validation_split], tfrecords_paths[cfg.training.validation_split:]
            logging.info("Using %s for validation", ",".join(validation_tfrecords_paths))

        _make_paths_absolute(cfg, ["model.model_path", "training.log_dir", "training.checkpoint_dir"])

        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0], pileup_image_channels=cfg.pileup.image_channels)
        model = hydra.utils.instantiate(cfg.model, image_shape, replicates, model_path=cfg.model.model_path, weights=cfg.training.initial_weights, base_trainable=cfg.training.initial_weights is None or cfg.training.base_trainable)

        dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True, num_parallel_reads=cfg.threads, pileup_image_channels=cfg.pileup.image_channels)
        validation_dataset = load_example_dataset(validation_tfrecords_paths, with_label=True, with_simulations=True, num_parallel_reads=cfg.threads, pileup_image_channels=cfg.pileup.image_channels) if validation_tfrecords_paths else None
        model.fit(cfg, dataset, validation_dataset=validation_dataset)
    
        model_path = os.path.join(os.getcwd(), "model.h5")
        logging.info("Saving model in: %s", model_path)
        model.save(model_path)

    elif cfg.command == "genotype":
        from .sample import Sample, sample_name_from_bam
        from .variant import Variant
        import pysam
        import pysam.bcftools as bcftools
        from .utilities.vcf import index_variant_file, bcftools_format
        from .genotyping import genotype_vcf
        import pysam.bcftools as bcftools

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
       
        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "genotypes.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        # model_path can be a single file, list, or dictionary keyed by variant type. For specific types
        # split out variants into separate files
        if isinstance(cfg.model.model_path, (dict, DictConfig)):
            # Split out variants by type
            logging.info("Splitting input VCF into %s SVs. All other variants types will be skipped", ",".join(cfg.model.model_path.keys()))
            split_input_dir = tempfile.mkdtemp()
            with pysam.VariantFile(hydra.utils.to_absolute_path(cfg.input)) as src_vcf_file:
                split_input = {kind: pysam.VariantFile(os.path.join(split_input_dir, f"{kind}.vcf.gz"), mode="wz", header=src_vcf_file.header) for kind in cfg.model.model_path}
                for record in src_vcf_file:
                    variant = Variant.from_pysam(record)
                    split_file = split_input.get(variant.type)
                    if split_file:  # TODO: Replace with walrus operator
                        split_file.write(record)
                for split_file in split_input.values():
                    split_file.close()
                    index_variant_file(_normalize_fname(split_file.filename))

            # Create corresponding model files
            type_input = {kind : (_normalize_fname(split_input[kind].filename), os.path.join(split_input_dir, f"{kind}.genotypes.vcf.gz"), model_path) for kind, model_path in cfg.model.model_path.items()}
        else:
            # Make sure input path is absolute
            type_input = {"ALL": (hydra.utils.to_absolute_path(cfg.input), output, cfg.model.model_path)}

        # Make sure other paths are absolute
        _make_paths_absolute(cfg, ["pileup.snv_vcf_input", "cache_dir"])
        
        # Create cache directory if it doesn't exists, otherwise get_file won't use it
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, mode=0o775, exist_ok=True)

        for kind, (input_path, output_path, model_path) in type_input.items():
            logging.info("Genotyping %s variants", kind)
            model_paths = [_get_file(cfg, path_or_url) for path_or_url in _as_list(model_path)]

            # Since setting the models may change the type of the configuration, we perform the merge here to override the type
            local_cfg = cfg.copy()
            OmegaConf.update(local_cfg, "model.model_path", model_paths, merge=False)

            genotype_vcf(
                local_cfg,
                input_path,
                samples,
                model_paths,
                output_path,
                progress_bar=True,
                evaluate=cfg.concordance,
                region=cfg.region,
            )
    
        # Merge and cleanup type specific files
        if isinstance(cfg.model.model_path, (dict, DictConfig)):
            logging.info("Merging output files into %s", output)
            bcftools.concat("-O", bcftools_format(output), "-o", output, "--allow-overlaps", *[type_path for _, type_path, _ in type_input.values()], catch_stdout=False)
            index_variant_file(output)
            shutil.rmtree(split_input_dir)

    elif cfg.command == "propose":
        from .propose.propose import propose_vcf

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "propose.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        propose_vcf(cfg, hydra.utils.to_absolute_path(cfg.input), output, hydra.utils.to_absolute_path(cfg.refine.simple_repeats_path), progress_bar=True)
    
    elif cfg.command == "refine":
        from .propose.refine import refine_vcf
        if cfg.refine.select_algo == "ml" and not OmegaConf.is_missing(cfg, "refine.classifier_path"):
            classifier_path = [hydra.utils.to_absolute_path(path) for path in _as_list(cfg.refine.classifier_path)]
        else:
            classifier_path = []
        
        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "genotypes.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)
        
        refine_vcf(cfg, hydra.utils.to_absolute_path(cfg.input), output, classifier_path=classifier_path, progress_bar=True)

    elif cfg.command == "preprocess":
        from .sample import compute_read_stats

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "stats.json"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["reference"])

        stats = compute_read_stats(cfg, hydra.utils.to_absolute_path(cfg.reads))
        with open(output, "w") as file:
            json.dump(stats, file)

    elif cfg.command == "filter":
        from .propose.filter import filter_vcf

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "filter.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        filter_vcf(cfg, hydra.utils.to_absolute_path(cfg.input), output, progress_bar=True)

if __name__ == "__main__":
    main()
