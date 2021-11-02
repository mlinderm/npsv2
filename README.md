# NPSV2: Non-parametric Structural Variant Genotyper

NPSV2 is a prototype Python-based tool for stand-alone genotyping of previously detected/reported deletion structural variants (SVs) in short-read whole genome sequencing (WGS) data. NPSV2 is the successor to the [NSPV SV genotyper](https://github.com/mlinderm/npsv). NPSV2 implements a deep learning-based approach for SV genotyping that employs NGS simulation to model the combined effects of the genomic region, sequencer and alignment pipeline.

NPSV2 is a work in progress that is currently under active development.

## Installation

When cloning NPSV2, make sure to recursively clone all of the submodules, i.e. `git clone --recursive git@github.com:mlinderm/npsv2.git`.

NPSV requires Python 3.6+ and a suite of command-line genomics tools. For convenience, a Docker file is provided that installs all of the dependencies. To build that image:
```
docker build -t npsv2 .
```

### Manual installation

To manually install and run NPSV from the source, you will need the following dependencies:

* ART (NGS simulator)
* bwa
* bedtools
* bcftools
* goleft
* htslib (i.e., tabix and bgzip)
* samblaster
* sambamba
* samtools

along with standard command-line utilities, CMake and a C++14 compiler. After installing the dependencies listed above, install the Python dependencies, and then NPSV itself via:
```
python3 -m pip install -r requirements.txt
python3 setup.py install
```

## Running NPSV2

NPSV2 requires basic information about the aligned reads (i.e. sequencer model, coverage, insert size distribution). These data are currently provided via a JSON-formatted stats file. The current prototype relies on the preprocessing tools built into NPSV.

### Running the NPSV2 tools with Docker

Given the multi-step workflow, the typical approach when using the Docker image is to run NPSV2 from a shell. The following command will start a Bash session in the Docker container (replace `/path/to/reference/directory` with the path to directory containing the reference genome and associated BWA indices). NPSV is most efficient when the BWA indices are loaded into shared memory. To load BWA indices into shared memory you will need to configure the Docker container with at least 10G of memory and set the shared memory size to 8G or more.

```
docker run --entrypoint /bin/bash \
    --shm-size=8g \
    -v `pwd`:/opt/npsv2 \
    -v /path/to/reference/directory:/data \
    -w /opt/npsv2 \
    -it \
    npsv2
```

## NPSV2 Genotyping

The NPSV package installs the `npsv2` executable, which executes the different commands in the genotyping workflow. NPSV2 uses [hydra](https://hydra.cc) to manage program configuration; all arguments/options are specified as hydra overrides.

### Prerequisites

NPSV2 requires the reference genome and these examples, in particular, require the "b37" reference.

### Basic Workflow

The minimal NPSV2 workflow requires the putative SV(s) as a VCF file, the aligned reads and basic sequencing statistics (the sequencer model, read length, the mean and SD of the insert size, and depth), and a previously trained network model. The following assumes you have copied a suitable model to `tests/results/model.h5`. A minimal example follows.

To run NPSV2 genotyping:

```
npsv2 command=genotype \
    reference=/data/human_g1k_v37.fasta \
    model.model_path=tests/results/model.h5 \
    input=tests/data/12_22129565_22130387_DEL.vcf.gz \
    reads=tests/data/12_22127565_22132387.bam \
    stats_path=tests/data/stats.json \
    output=tests/results/12_22129565_22130387_DEL.npsv2.vcf.gz \
    load_reference=true
```

This will produce a VCF file `tests/results/12_22129565_22130387_DEL.npsv2.vcf.gz` (determined by the output parameter) with the genotypes. The input variant is derived from the Genome-in-a-Bottle SV dataset; NPSV successfully genotypes this variant as homozygous alternate.

The `load_reference=true` argument will automatically load the BWA index into shared memory (and cleanup after completion) if it has not already been loaded. 

