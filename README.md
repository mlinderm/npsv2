# NPSV-deep: A deep learning method for genotyping structural variants in genome sequencing data

NPSV-deep is a Python-based tool for stand-alone genotyping of previously detected/reported deletion (DEL) and insertion (INS) structural variants (SVs) in short-read genome sequencing (SRS) data. NPSV-deep is the successor to the [NSPV SV genotyper](https://github.com/mlinderm/npsv). NPSV-deep implements a deep learning-based approach for SV genotyping that employs SRS simulation to model the combined effects of the genomic region, sequencer and alignment pipeline.

NPSV-deep is a work in progress that is currently under active development.

## Installation

When cloning NPSV-deep, make sure to recursively clone all of the submodules, i.e. `git clone --recursive git@github.com:mlinderm/npsv2.git`.

NPSV-deep requires Python 3.8+ and a suite of command-line genomics tools. For convenience, a Docker file is provided that installs all of the dependencies. To build that image:
```
docker build -t npsv2 .
```

### Manual installation

To manually install and run NPSV-deep from the source, you will need the following dependencies:

* ART (NGS simulator)
* bwa
* bedtools
* bcftools
* goleft
* htslib (i.e., tabix and bgzip)
* jellyfish (with Python bindings)
* samblaster
* sambamba
* samtools

along with standard command-line utilities, CMake and a C++14 compiler. After installing the dependencies listed above, install the Python dependencies, and then NPSV itself via:
```
python3 -m pip install -r requirements.txt
python3 setup.py install
```

## Running NPSV-deep

NPSV-deep requires basic information about the aligned reads (i.e. sequencer model, coverage, insert size distribution). These data are currently provided via a JSON-formatted stats file. You can generate that file with the provided `preprocess` command.

### Running the NPSV-deep tools with Docker

Given the multi-step workflow, the typical approach when using the Docker image is to run NPSV-deep from a shell. The following command will start a Bash session in the Docker container (replace `/path/to/reference/directory` with the path to directory containing the reference genome and associated BWA indices). NPSV-deep is most efficient when the BWA indices are loaded into shared memory. To load BWA indices into shared memory you will need to configure the Docker container with at least 12G of memory and set the shared memory size to 8G or more.

```
docker run --entrypoint /bin/bash \
    --shm-size=8g \
    -v `pwd`:/opt/npsv2 \
    -v /path/to/reference/directory:/data \
    -w /opt/npsv2 \
    -it \
    npsv2
```

## NPSV-deep Genotyping

The NPSV-deep package installs the `npsv2` executable, which executes the different commands in the genotyping workflow. NPSV-deep uses [hydra](https://hydra.cc) to manage program configuration; all arguments/options are specified as hydra overrides.

### Prerequisites

NPSV-deep requires the reference genome and these examples, in particular, require the "b37" reference.

### Basic Workflow

The minimal NPSV-deep workflow requires the putative SV(s) as a VCF file, the aligned reads and basic sequencing statistics (the sequencer model, read length, the mean and SD of the insert size, and depth), and a previously trained network model. The following assumes you have copied a suitable model to `tests/results/model.h5`. A minimal example follows.

To run NPSV-deep genotyping:

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

### Preprocessing to create a "stats" file

NPSV-deep utilizes information about the aligned reads to inform simulation and image generation. The preprocessing step, run with the preprocess sub-command for `npsv2`, will create a JSON file with the relevant stats. Note that since this example BAM file only includes reads in a small region on chromosome 12, the results for the following example command will not be meaningful.

```
npsv2 command=preprocess \
    reference=/data/human_g1k_v37.fasta \
    sequencer=HSXn \
    reads=tests/data/12_22127565_22132387.bam \
    output=tests/results/stats.json
```

The `sequencer` argument specifies the sequencer model and thus the profile to use with the [ART NGS simulator](https://www.niehs.nih.gov/research/resources/software/biostatistics/art/index.cfm). Currently available profiles in ART are:

```
GA1 - GenomeAnalyzer I (36bp,44bp), GA2 - GenomeAnalyzer II (50bp, 75bp)
HS10 - HiSeq 1000 (100bp),          HS20 - HiSeq 2000 (100bp),      HS25 - HiSeq 2500 (125bp, 150bp)
HSXn - HiSeqX PCR free (150bp),     HSXt - HiSeqX TruSeq (150bp),   MinS - MiniSeq TruSeq (50bp)
MSv1 - MiSeq v1 (250bp),            MSv3 - MiSeq v3 (250bp),        NS50 - NextSeq500 v2 (75bp)
```

Preprocessing is multi-threaded. Specifying multiple threads, e.g. `threads=8`, will improve preprocessing performance.

## Developing on an Apple M1

An experimental Dockerfile is including for testing on systems with an Apple M1. To build an Apple-specific container:
```
docker build -f Dockerfile.m1 -t npsv2-m1 .
```
and then start the npsv2-m1 container.
