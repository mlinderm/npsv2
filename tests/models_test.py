import argparse, os, tempfile, unittest
from unittest.mock import patch, call

import numpy as np
import hydra
from hydra.experimental import compose, initialize

from npsv2.models import SimulatedEmbeddingsModel, JointEmbeddingsModel
#from npsv2.models import TripletModel, JointEmbeddingsModel, WindowedJointEmbeddingsModel
from npsv2.images import load_example_dataset, vcf_to_tfrecords, _extract_metadata_from_first_example
from npsv2.sample import Sample
from .images_test import _mock_chunk_genome, _mock_reference_sequence, _mock_simulate_variant_sequencing

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

def setUpModule():
    initialize(config_path="../src/npsv2/conf")

def tearDownModule():
    hydra.core.global_hydra.GlobalHydra.instance().clear()

#@unittest.skip("Development only")
class SimulatedEmbeddingsModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = compose(config_name="config", overrides=[
            "training.epochs=1",
            "model=simulated_embeddings"
        ])

    def test_configuration_overrides(self):
        self.assertEqual(self.cfg.training.variants_per_batch, 9)

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test.tfrecords.gz")), "No test inputs available")
    def test_construct_model(self):
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 5), 5)
        self.assertIsInstance(model, SimulatedEmbeddingsModel)
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


@unittest.skip("Development only")
class JointEmbeddingsModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = compose(config_name="config", overrides=[
            "training.epochs=1",
            "model=joint_embeddings"
        ])
    
    def test_construct_model(self):
        model = hydra.utils.instantiate(self.cfg.model, (100, 300, 5), 5)
        self.assertIsInstance(model, JointEmbeddingsModel)
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


@unittest.skip("Development only")
class TripletModelTest(unittest.TestCase):
    def test_construct_model(self):
        model = TripletModel((100, 300, 5), 5)
        model.summary()

    def test_fit_model(self):
        model = TripletModel((100, 300, 5), 5)
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test.tfrecords.gz"), with_simulations=True, with_label=True
        )

        model.fit(dataset, epochs=1)
    
    def test_predict_model(self):
        model = TripletModel((100, 300, 5), 5)
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test.tfrecords.gz"), with_simulations=True, with_label=True
        )

        genotypes, *_ = model.predict(dataset)



@unittest.skip("Development only")
class WindowedJointEmbeddingsModelTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(
            tempdir=self.tempdir.name,
            reference=None,
            flank=1000,
            replicates=2,
            threads=1,
            sample_ref=False,
            exclude_bed=None,
            augment=False,
        )
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_construct_model(self):
        model = WindowedJointEmbeddingsModel((100, 50, 5), 5)
        model.summary()

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test-windows.tfrecord.gz")), "No test inputs available")
    def test_fit_model(self):
        # To generate dataset
        # npsv2 examples -r /data/human_g1k_v37.fasta -i tests/data/1_899922_899992_DEL.vcf.gz -b tests/data/1_896922_902998.bam -o tests/data/test-windows.tfrecord.gz --stats-path tests/data/stats.json --replicates 2 -s HG002 --windowed
        
        model = WindowedJointEmbeddingsModel((100, 50, 5), 2)  
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test-windows.tfrecord.gz"), with_simulations=True, with_label=True
        )

        model.fit(dataset, epochs=1)

    @unittest.skipUnless(os.path.exists(os.path.join(FILE_DIR, "test-windows.tfrecord.gz")), "No test inputs available")
    def test_predict_model(self):
        model = WindowedJointEmbeddingsModel((100, 50, 5))  
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test-windows.tfrecord.gz"), with_simulations=True, with_label=True
        )

        genotypes, distances, *_ = model.predict(dataset)
        self.assertEqual(np.argmax(genotypes), np.argmin(distances))