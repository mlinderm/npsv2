#!/usr/bin/env python3
import argparse, json, logging, os, random, re, subprocess, sys, tempfile, typing
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


# Resolvers for use with Hydra
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("swap_ext", lambda path, old_ext, new_ext: re.sub(old_ext + "$", new_ext, path))
OmegaConf.register_new_resolver("strip_ext", lambda path: os.path.splitext(path)[0])
OmegaConf.register_new_resolver("escape", lambda path: str(path).replace("[","").replace("]","").replace(", ","_")) 

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), file=sys.stderr)
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
            split_index = int(len(tfrecords_paths) * cfg.training.validation_split)
            logging.info("Using random selection of %d input files for validation", split_index)
            validation_tfrecords_paths, tfrecords_paths = tfrecords_paths[:split_index], tfrecords_paths[split_index:]
            logging.info("Using %s for validation", ",".join(validation_tfrecords_paths))

        _make_paths_absolute(cfg, ["model.model_path", "training.log_dir", "training.checkpoint_dir"])

        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
        model = hydra.utils.instantiate(cfg.model, image_shape, replicates, model_path=cfg.model.model_path, weights=cfg.training.initial_weights)

        dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True, num_parallel_reads=cfg.threads)
        validation_dataset = load_example_dataset(validation_tfrecords_paths, with_label=True, with_simulations=True, num_parallel_reads=cfg.threads) if validation_tfrecords_paths else None
        model.fit(cfg, dataset, validation_dataset=validation_dataset)
    
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

        _make_paths_absolute(cfg, ["model.model_path"])

        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
        model = hydra.utils.instantiate(cfg.model, image_shape, 1)

        errors = 0
        rows = []
        for features, original_label in load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True):
            #print(features)
            if original_label is None:
                continue  # Skip invalid genotypes
                
            # Extract metadata for the variant
            variant_proto = npsv2_pb2.StructuralVariant.FromString(features.pop("variant/encoded").numpy())
            
            # Predict genotype
            dataset = tf.data.Dataset.from_tensors((features, original_label))
            genotypes, distances, *_  = model.predict(cfg, dataset)

            # if tf.math.argmax(genotypes, axis=1) != original_label.numpy() and original_label.numpy() == 0:
            # #if tf.math.argmax(genotypes, axis=1) == 0 and  original_label.numpy() == 1:
            #     print(variant_proto, genotypes, distances)
            #     errors += 1
            # if errors == 10:
            #     break

            # if tf.math.argmax(genotypes, axis=1) != original_label:
            #     print(variant_proto, genotypes)

            # Construct the DataFrame rows
            rows.append(pd.DataFrame({
                "SVLEN": variant_proto.svlen,
                "LABEL": original_label.numpy(),
                "AC": tf.math.argmax(genotypes, axis=1),
            }))

        table = pd.concat(rows, ignore_index=True)
        table["AC"] = pd.Categorical(table["AC"], categories=[0, 1, 2])
        table["LABEL"] = pd.Categorical(table["LABEL"], categories=[0, 1, 2])
        table["MATCH"] = table.LABEL == table.AC

        # Print various metrics
        confusion_matrix = pd.crosstab(table.LABEL, table.AC, rownames=["Truth"], colnames=["Test"], margins=True, dropna=False)

        gt_conc = (confusion_matrix.loc[0, 0] + confusion_matrix.loc[1, 1] + confusion_matrix.loc[2, 2]) / confusion_matrix.loc["All", "All"] #np.mean(table.MATCH)
        nr_conc = (confusion_matrix.loc[0, 0] + confusion_matrix.loc[[1, 2], [1, 2]].to_numpy().sum()) / confusion_matrix.loc["All", "All"] # np.mean((table.LABEL != 0) == (table.AC != 0))
        print(f"Accuracy - Genotype concordance: {gt_conc:.3}, Non-reference Concordance: {nr_conc:.3}")

        tp = confusion_matrix.loc[[1, 2], [1, 2]].to_numpy().sum()
        fp = confusion_matrix.loc[[0], [1, 2]].to_numpy().sum()
        fn = confusion_matrix.loc[[1, 2], [0]].to_numpy().sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"Event classification - Precision: {precision:.3}, Recall: {recall:.3}, F1: {f1:.3}")
        
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
       
        # Make sure model path (and other paths) are absolute
        model_paths = [hydra.utils.to_absolute_path(path) for path in _as_list(cfg.model.model_path)]
        _make_paths_absolute(cfg, ["pileup.snv_vcf_input"])

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "genotypes.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        genotype_vcf(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            samples,
            model_paths,
            output,
            progress_bar=True,
            evaluate=cfg.concordance,
            region=cfg.region,
        )
    
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


if __name__ == "__main__":
    main()
