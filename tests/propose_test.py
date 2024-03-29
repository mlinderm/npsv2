import os, subprocess, tempfile, unittest, warnings
import pysam
import ray
import hydra
from omegaconf import OmegaConf
from parameterized import parameterized
from npsv2.propose import propose, refine, filter

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


def setUpModule():
     # Ignore resource warnings within Ray
    warnings.simplefilter("ignore", ResourceWarning)
    ray.init(num_cpus=1, num_gpus=0, local_mode=True, include_dashboard=False)

    hydra.initialize(config_path="../src/npsv2/conf")


def tearDownModule():
    ray.shutdown()
    hydra.core.global_hydra.GlobalHydra.instance().clear()


class ProposeTestSuite(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=["reference=/data/human_g1k_v37.fasta"],)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    @unittest.skipUnless(
        os.path.exists("/data/human_g1k_v37.fasta") and os.path.exists("/data/simple_repeats.b37.bed.gz"),
        "Reference genome not available",
    )
    def test_propose(self):
        output_path = os.path.join(self.tempdir.name, "test.vcf")
        propose.propose_vcf(
            self.cfg, os.path.join(FILE_DIR, "propose_input.vcf"), output_path, "/data/simple_repeats.b37.bed.gz"
        )
        self.assertTrue(os.path.exists(output_path))


class RefineTestSuite(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(config_name="config", overrides=[])
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_generate_rows_from_record(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "refine_input.vcf")))
        table = refine._record_to_rows(record, [0.1])
        self.assertEqual(table.shape[0], 1)  # VCF has only one sample (thus one row)

    @parameterized.expand([("original",), ("ml",), ("min_distance",), ("max_prob",)])
    def test_refine(self, select_algo):
        cfg = OmegaConf.merge(self.cfg, {"refine": {"select_algo": select_algo}})
        output_path = os.path.join(self.tempdir.name, "test.vcf")
        refine.refine_vcf(
            cfg,
            os.path.join(FILE_DIR, "refine_input.vcf"),
            output_path,
            classifier_path=[os.path.join(FILE_DIR, "refineML_model.joblib")],
        )
        self.assertTrue(os.path.exists(output_path))


class RefineTrainTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_train_refine_model(self):
        model_path = os.path.join(self.tempdir.name, "test.joblib")
        refine.train_model(
            os.path.join(FILE_DIR, "refine_input.vcf"), os.path.join(FILE_DIR, "refine_pbsv.tsv"), model_path
        )
        self.assertTrue(os.path.exists(model_path))


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta"), "Reference genome not available",
)
class FilterTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "reference=/data/human_g1k_v37.fasta",
                f"refine.reference_jf_path={os.path.join(self.tempdir.name, 'reference.jf')}",
                f"refine.reads_jf_path={os.path.join(self.tempdir.name, 'reads.jf')}",
            ],
        )

        subprocess.check_call(
            f"jellyfish count -s 10000 -m {self.cfg.refine.filterk} -o {self.cfg.refine.reference_jf_path} {os.path.join(FILE_DIR, '1_896922_902998.fasta')}",
            shell=True,
        )
        subprocess.check_call(
            f"jellyfish count -s 100000 -m {self.cfg.refine.filterk} -o {self.cfg.refine.reads_jf_path} <(samtools fastq {os.path.join(FILE_DIR, '1_896922_902998.bam')})",
            shell=True,
            executable="/bin/bash",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_kmer_filtering(self):
        output_path = os.path.join(self.tempdir.name, "test.vcf")
        filter.filter_vcf(self.cfg, os.path.join(FILE_DIR, "refine_input.vcf"), output_path)
        self.assertTrue(os.path.exists(output_path))

