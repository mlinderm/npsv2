#!/usr/bin/env python3
import argparse, logging, os, subprocess, sys, tempfile
from omegaconf import DictConfig, OmegaConf, ListConfig
import hydra
import tensorflow as tf
from tqdm import tqdm
from .simulation import bwa_index_loaded

def _as_list(item_or_list):
    """Convert scalar argument to list, or pass list through"""
    return item_or_list if isinstance(item_or_list, (list, ListConfig)) else [item_or_list]

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
    if cfg.simulation.replicates > 0:
        cfg.shared_reference = bwa_index_loaded(hydra.utils.to_absolute_path(cfg.reference), load=cfg.load_reference)
        if not cfg.shared_reference:
            logging.warning(
                "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
                cfg.reference,
            )


def _is_tfrecords_file(filename: str) -> bool:
    return filename.endswith((".tfrecords", "tfrecords.gz"))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), file=sys.stderr)
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
                example_to_image(cfg, example, image_path, with_simulations=True, max_replicates=cfg.simulation.replicates)
        else:  # Assume it is a VCF file
            _check_shared_reference(cfg)
            sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

            examples = make_vcf_examples(cfg, input_path, hydra.utils.to_absolute_path(cfg.reads), sample, simulate=True)
            for i, example in enumerate(tqdm(examples, desc="Generating images for each variant")):
                image_path = os.path.join(os.getcwd(), f"variant{i}.png")
                example_to_image(cfg, example, image_path, with_simulations=True, max_replicates=cfg.simulation.replicates)

    elif cfg.command == "train":
        from .images import _extract_metadata_from_first_example, load_example_dataset
        
        _configure_gpu()

        tfrecords_paths = [cfg.input] if isinstance(cfg.input, str) else cfg.input
        tfrecords_paths = [hydra.utils.to_absolute_path(p) for p in tfrecords_paths]
        
        image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
        model = hydra.utils.instantiate(cfg.model, image_shape, replicates)

        dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True, num_parallel_reads=cfg.threads)
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

        errors = 0
        rows = []
        for features, original_label in load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True):
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
       
        # Make sure model path is absolute
        cfg.model.model_path = hydra.utils.to_absolute_path(cfg.model.model_path)

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "genotypes.vcf.gz"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        genotype_vcf(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            samples,
            output,
            progress_bar=True,
        )
    
    elif cfg.command == "propose":
        from .propose.propose import propose_vcf

        propose_vcf(cfg, hydra.utils.to_absolute_path(cfg.input), hydra.utils.to_absolute_path(cfg.output), hydra.utils.to_absolute_path(cfg.refine.simple_repeats_path), progress_bar=True)
    
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

if __name__ == "__main__":
    main()
