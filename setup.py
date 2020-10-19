from setuptools import setup, find_packages
from setuptools import Command
from distutils.spawn import find_executable
import os, subprocess, sys

# Find the Protocol Buffer compiler (protoc)
if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
    protoc = os.environ["PROTOC"]
else:
    protoc = find_executable("protoc")

def generate_proto(source: str):
    """Invoke Protocol Buffer compiler if .proto file is newer than generated code

    Args:
        source (str): Path to .proto file
    """
    # Adapted from Chromium
    output = source.replace(".proto", "_pb2.py")
    if not os.path.exists(output) or (
        os.path.exists(source) and os.path.getmtime(source) > os.path.getmtime(output)
    ):
        if not os.path.exists(source):
            sys.stderr.write("Can't find required file: %s\n" % source)
            sys.exit(-1)
        if protoc == None:
            sys.stderr.write(
                "protoc is not installed nor found in ../src.  Please compile it "
                "or install the binary package.\n"
            )
            sys.exit(-1)
        protoc_command = [protoc, "-I.", "--python_out=.", source]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)


class BuildProtobuf(Command):
    description = "Compile protobuf descriptions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        generate_proto("src/npsv2/npsv2.proto")


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="npsv2",
    version="0.1.0",
    description="Non-parametric genotyper for structural variants",
    long_description=readme,
    author="Michael Linderman",
    author_email="mlinderman@middlebury.edu",
    license=license,
    url="https://github.com/mlinderm/npsv2",
    scripts=[],
    entry_points="""
        [console_scripts]
        npsv2=npsv2.main:main
    """,
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    data_files=[
        (
            "etc",
            [
                "etc/human_g1k_v37.genome",
                "etc/human_g1k_v37.gaps.bed.gz",
                "etc/human_g1k_v37.gaps.bed.gz.tbi",
            ],
        )
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=["numpy", "pysam", "Pillow", "tqdm"],
    extras_require={"tf": ["tensorflow"], "tf_gpu": ["tensorflow-gpu"],},
    cmdclass={"protobuf": BuildProtobuf},
)
