import os, unittest

from npsv2.models import TripletModel, JointEmbeddingsModel
from npsv2.images import load_example_dataset

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

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
class JointEmbeddingsModelTest(unittest.TestCase):
    def test_construct_model(self):
        model = JointEmbeddingsModel((100, 300, 5), 5)
        model.summary()

    def test_fit_model(self):
        model = JointEmbeddingsModel((100, 300, 5), 5)
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test.tfrecords.gz"), with_simulations=True, with_label=True
        )

        model.fit(dataset, epochs=1)

    def test_predict_model(self):
        model = JointEmbeddingsModel((100, 300, 5), 5)
        dataset = load_example_dataset(
            os.path.join(FILE_DIR, "test.tfrecords.gz"), with_simulations=True, with_label=True
        )

        genotypes, *_ = model.predict(dataset)