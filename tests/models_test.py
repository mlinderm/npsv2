import argparse, os, tempfile, unittest
from unittest.mock import patch, call

import numpy as np
import hydra
from omegaconf import OmegaConf

from npsv2 import models
from npsv2.images import load_example_dataset, vcf_to_tfrecords, _extract_metadata_from_first_example
from npsv2.sample import Sample
from .images_test import _mock_reference_sequence, _mock_simulate_variant_sequencing

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

def setUpModule():
    hydra.initialize(config_path="../src/npsv2/conf")

def tearDownModule():
    hydra.core.global_hydra.GlobalHydra.instance().clear()


class EncoderTest(unittest.TestCase):
    def test_original_encoder(self): 
        encoder = models._contrastive_encoder(
            (100, 300, 5),
            normalize_embedding=False,
            projection_size=[512],
            normalize_projection=True,
            batch_normalize_projection=True
        )
        self.assertEqual(encoder.name, "encoder")
        encoder.summary()
        embeddings, _, projections = encoder.output_shape
        self.assertEqual(embeddings, (None, 2048))
        self.assertEqual(projections, (None, 512))
    
    def test_supcon_network_with_linear_projection(self):
        encoder = models._contrastive_encoder(
            (100, 300, 5),
            normalize_embedding=True,
            stop_gradient_before_projection=False,
            projection_size=[128],
            normalize_projection=True,
            batch_normalize_projection=False
        )
        self.assertEqual(encoder.name, "encoder")

        embeddings, normalized_embeddings, projections = encoder.output_shape
        self.assertEqual(embeddings, (None, 2048))
        self.assertEqual(projections, (None, 128))

    def test_supcon_network_with_MLP(self):
        encoder = models._contrastive_encoder(
            (100, 300, 5),
            normalize_embedding=True,
            stop_gradient_before_projection=False,
            projection_size=[2048, 128],
            normalize_projection=True,
            batch_normalize_projection=False
        )
        self.assertEqual(encoder.name, "encoder")

        embeddings, normalized_embeddings, projections = encoder.output_shape
        self.assertEqual(embeddings, (None, 2048))
        self.assertEqual(projections, (None, 128))

    def test_base_model_only(self):
        encoder = models._contrastive_encoder(
            (100, 300, 5),
            projection_size=[],
        )
        self.assertEqual(encoder.name, "encoder")

        embeddings, *_ = encoder.output_shape
        self.assertEqual(embeddings, (None, 2048))


    def test_type_specific(self):
        encoder = models._contrastive_encoder(
            (100, 300, 7),
            normalize_embedding=False,
            projection_size=[512],
            normalize_projection=True,
            batch_normalize_projection=True,
            typed_projection=True,
        )
        self.assertEqual(encoder.name, "encoder")
        encoder.summary()


@unittest.skip("Development only")
class SupervisedBaselineModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=[
            "training.epochs=1",
            "model=supervised_baseline"
        ])
    
    def test_construct_model(self):
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 5), 5)
        self.assertIsInstance(model, models.SupervisedBaselineModel)
        model._model.get_layer("encoder")  # Will raise if 'encoder' is not defined
        model.summary()

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_fit_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        
        model = hydra.utils.instantiate(self.cfg.model, image_shape, replicates)
        dataset = load_example_dataset(dataset_path, with_simulations=False, with_label=True)  
        model.fit(self.cfg, dataset)

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_predict_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        
        model = hydra.utils.instantiate(self.cfg.model, image_shape, 1)
        dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)      
        genotypes, *_ = model.predict(self.cfg, dataset)


@unittest.skip("Development only")
class SimulatedEmbeddingsModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=[
            "training.epochs=1",
            "model=simulated_embeddings"
        ])

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_construct_model(self):
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 5), 5)
        self.assertIsInstance(model, models.SimulatedEmbeddingsModel)
        model.summary()

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_fit_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        
        model = hydra.utils.instantiate(self.cfg.model, image_shape, replicates)
        dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)  
        model.fit(self.cfg, dataset)

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_predict_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        
        model = hydra.utils.instantiate(self.cfg.model, image_shape, 1)
        dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)      
        genotypes, *_ = model.predict(self.cfg, dataset)


#@unittest.skip("Development only")
class JointEmbeddingsModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=[
            "training.epochs=1",
            "model=joint_embeddings",
            "training.contrastive_margin=0.5",
        ])
    
    def test_construct_model(self):
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 5), 5)
        self.assertIsInstance(model, models.JointEmbeddingsModel)
        model._model.get_layer("encoder").summary()  # Will raise if 'encoder' is not defined
        model.summary()
        
        genotypes_shape, distances_shape, query_embeddings_shape, support_embeddings_shape = model._model.output_shape
        self.assertEqual(genotypes_shape, (None, 3))
        self.assertEqual(distances_shape, (None, 3))
        projection, *_ = self.cfg.model.projection_size
        self.assertEqual(query_embeddings_shape, (None, 1, projection))
        self.assertEqual(support_embeddings_shape, (None, 1, 3, projection))


    def test_construct_typed_projection_model(self):
        cfg = OmegaConf.merge(self.cfg, {"model": {"typed_projection": True}})
        model = hydra.utils.instantiate(cfg.model, (100, 300, 7), 5)
        self.assertIsInstance(model, models.JointEmbeddingsModel)
        model._model.get_layer("encoder")  # Will raise if 'encoder' is not defined
        model.summary()

    @unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(__file__), "results", "model.h5")), "No model weights available")
    def test_construct_ensemble_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "results", "model.h5")
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 7), 5, model_path=[model_path]*2)
        model.summary()

        genotypes_shape, distances_shape, query_embeddings_shape, support_embeddings_shape = model._model.output_shape
        self.assertEqual(genotypes_shape, (None, 3))
        self.assertEqual(distances_shape, (None, 3))
        projection, *_ = self.cfg.model.projection_size
        self.assertEqual(query_embeddings_shape, (None, 2, projection))
        self.assertEqual(support_embeddings_shape, (None, 2, 3, projection))

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_fit_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        self.assertGreater(replicates, 0)
        model = hydra.utils.instantiate(self.cfg.model, image_shape, replicates)
        dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)
        validation_dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)
        model.fit(self.cfg, dataset, validation_dataset=validation_dataset)

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_predict_model(self):
        dataset_path = os.path.join(FILE_DIR, "test.tfrecords.gz")
        image_shape, replicates = _extract_metadata_from_first_example(dataset_path)
        
        model = hydra.utils.instantiate(self.cfg.model, image_shape, 1)
        dataset = load_example_dataset(dataset_path, with_simulations=True, with_label=True)
        genotypes, *_ = model.predict(self.cfg, dataset)

