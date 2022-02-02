import os, unittest
from npsv2.sample import Sample

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

class SampleLoadTestSuite(unittest.TestCase):
    def test_load_stats_json(self):
        json_path = os.path.join(FILE_DIR, "stats.json")
        sample = Sample.from_json(json_path)

        self.assertEqual(sample.read_length, 148)
        self.assertAlmostEqual(sample.chrom_mean_coverage("1"), 1.0230288573642952*sample.mean_coverage, places=4)