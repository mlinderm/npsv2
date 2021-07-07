import os, tempfile, unittest
import pysam
import hydra
from omegaconf import OmegaConf
from npsv2.propose import refine

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

def setUpModule():
    hydra.initialize(config_path="../src/npsv2/conf")

def tearDownModule():
    hydra.core.global_hydra.GlobalHydra.instance().clear()

class RefineFeatureTestSuite(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=[])
    
    def test_generate_rows_from_record(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "propose.vcf")))
        table = refine._record_to_rows(record)
        self.assertEqual(table.shape[0], 1) # VCF has only one sample (thus one row)

    def test_refine(self):
        refine.refine_vcf(self.cfg, os.path.join(FILE_DIR, "propose.vcf"), "/dev/stdout", classifier_path=os.path.join(FILE_DIR, 'refineML_model.joblib'))