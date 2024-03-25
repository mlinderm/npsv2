# NPSV-deep: A deep learning method for genotyping structural variants in genome sequencing data

NPSV-deep is a Python-based tool for stand-alone genotyping of previously detected/reported deletion (DEL) and insertion (INS) structural variants (SVs) in short-read genome sequencing (SRS) data. NPSV-deep is the successor to the [NSPV SV genotyper](https://github.com/mlinderm/npsv). NPSV-deep implements a deep learning-based approach for SV genotyping that employs SRS simulation to model the combined effects of the genomic region, sequencer and alignment pipeline.

NPSV-deep is described in the following publication:

Linderman MD, Wallace J, van der Heyde A, Wieman E, Brey D, Shi Y, Hansen P, Shamsi Z, Gelb BD, Bashir A. [NPSV-deep: A deep learning method for genotyping structural variants in short read genome sequencing data](https://doi.org/10.1093/bioinformatics/btae129). Bioinformatics. 2024;40(3).

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
* whatshap

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
    -v /path/to/reference/directory:/data \
    -w /opt/npsv2 \
    -it \
    npsv2
```

## NPSV-deep Genotyping

The NPSV-deep package installs the `npsv2` executable, which executes the different commands in the genotyping workflow. NPSV-deep uses [hydra](https://hydra.cc) to manage program configuration; all arguments/options are specified as hydra overrides.

### Prerequisites

NPSV-deep requires the reference genome and these examples, in particular, require the "b37" reference. To obtain and index those files from within the Docker container:

```plaintext
cd /data
curl ftp://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/technical/reference/human_g1k_v37.fasta.gz -o human_g1k_v37.fasta.gz
gunzip human_g1k_v37.fasta.gz
bwa index human_g1k_v37.fasta
samtools faidx human_g1k_v37.fasta
```

The following commands also assume you have create a `tests/results` directory

```plaintext
mkdir -p tests/results
```

### Basic Workflow

The minimal NPSV-deep workflow requires the putative SV(s) as a VCF file, the aligned reads and basic sequencing statistics (the sequencer model, read length, the mean and SD of the insert size, and depth), and a previously trained network model. The typical workflow also uses phased SNVs called in the same sample.

By default, NPSV-deep will automatically download and cache pre-trained models from a repository on [Hugging Face](https://huggingface.co/mlinderman/npsvdeep). Different models can be specified via the `model.model_path` configuration parameter.

To run NPSV-deep genotyping:

```plaintext
npsv2 command=genotype \
    reference=/data/human_g1k_v37.fasta \
    input=tests/data/12_22129565_22130387_DEL.vcf.gz \
    reads=tests/data/12_22127565_22132387.bam \
    stats_path=tests/data/stats.json \
    output=tests/results/12_22129565_22130387_DEL.npsv2.vcf.gz \
    load_reference=true
```

This will produce a VCF file `tests/results/12_22129565_22130387_DEL.npsv2.vcf.gz` (determined by the `output` parameter) with the genotypes. The input variant is derived from the Genome-in-a-Bottle SV dataset; NPSV successfully genotypes this variant as homozygous alternate. The genotype is determined from the minimum index of the `DS` field, the distances between the actual and simulated data for each possible genotype. Note that due to random variation in the simulation different runs will produce slightly different distances.

```plaintext
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
12	22129565	HG2_PB_SVrefine2Falcon2Bionano_6426	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	20	PASS	ClusterIDs=HG2_10X_SVrefine210Xhap12_9132:HG2_PB_PB10Xdip_7025:HG2_PB_PB10Xdip_7024:HG3_PB_pbsv_12731:HG4_PB_pbsv_13042:HG2_PB_pbsv_13047:HG3_PB_SVrefine2Falcon1Dovetail_7887:HG4_Ill_SVrefine2DISCOVARDovetail_9275:HG3_PB_SVrefine2PBcRDovetail_6143:HG3_PB_HySA_19630:HG2_PB_SVrefine2PBcRplusDovetail_2270:HG2_PB_SVrefine2PB10Xhap12_9410:HG2_PB_SVrefine2Falcon2Bionano_6426:HG2_PB_SVrefine2Falcon1plusDovetail_2387:HG2_Ill_SVrefine2DISCOVARplusDovetail_2630;NumClusterSVs=15;ExactMatchIDs=HG2_10X_SVrefine210Xhap12_9132:HG4_Ill_SVrefine2DISCOVARDovetail_9275:HG3_PB_SVrefine2PBcRDovetail_6143:HG3_PB_HySA_19630:HG2_PB_SVrefine2PBcRplusDovetail_2270:HG2_PB_SVrefine2PB10Xhap12_9410:HG2_PB_SVrefine2Falcon2Bionano_6426:HG2_PB_SVrefine2Falcon1plusDovetail_2387:HG2_Ill_SVrefine2DISCOVARplusDovetail_2630;NumExactMatchSVs=9;ClusterMaxShiftDist=0.00488102;ClusterMaxSizeDiff=0.00487211;ClusterMaxEditDist=0.00854179;PBcalls=12;Illcalls=2;TenXcalls=1;CGcalls=0;PBexactcalls=6;Illexactcalls=2;TenXexactcalls=1;CGexactcalls=0;HG2count=9;HG3count=4;HG4count=2;NumTechs=3;NumTechsExact=3;SVLEN=-822;DistBack=4675;DistForward=-689;DistMin=-689;DistMinlt1000=TRUE;MultiTech=TRUE;MultiTechExact=TRUE;SVTYPE=DEL;sizecat=300to999;DistPASSHG2gt49Minlt1000=FALSE;DistPASSMinlt1000=FALSE;MendelianError=FALSE;HG003_GT=1/1;HG004_GT=1/1;TRall=TRUE;TRgt100=TRUE;TRgt10k=FALSE;segdup=FALSE;REPTYPE=CONTRAC;BREAKSIMLENGTH=823;REFWIDENED=12:22129566-22131210	GT:DS:DHFFC:FS:SOR	1/1:0.9457,0.9968,0.0977:0.371:0:0.693
```

*Creating the simulated replicates is more efficient when the BWA indices are loaded into shared memory prior to running NPSV-deep (and thus doesn't need to re-loaded for each replicate).* The `load_reference=true` argument will automatically load the BWA index into shared memory (and cleanup after completion). If you are running multiple short commands, you can preload the index manually with `bwa shm /data/human_g1k_v37.fasta` (and omit the `load_reference` argument).

A more typical example incorporates phased SNVs into the pileup image generation via the `pileup.snv_vcf_input` parameter, e.g.,

```plaintext
npsv2 command=genotype \
    reference=/data/human_g1k_v37.fasta \
    input=tests/data/12_22129565_22130387_DEL.vcf.gz \
    reads=tests/data/12_22127565_22132387.bam \
    stats_path=tests/data/stats.json \
    pileup.snv_vcf_input=tests/data/12_22129565_22130387_DEL.snvs.vcf.gz \
    output=tests/results/12_22129565_22130387_DEL.npsv2.vcf.gz \
    load_reference=true
```

### Preprocessing to create a "stats" file

NPSV-deep utilizes information about the aligned reads to inform simulation and image generation. The preprocessing step, run with the preprocess sub-command for `npsv2`, will create a JSON file with the relevant stats. Note that since this example BAM file only includes reads in a small region on chromosome 12, the results for the following example command will not be meaningful.

```
npsv2 command=preprocess \
    reference=/data/human_g1k_v37.fasta \
    sequencer=HSXn \
    reads=tests/data/12_22127565_22132387.bam \
    output=tests/results/stats.json
```

The `sequencer` argument specifies the sequencer model and thus the profile to use with the [ART NGS simulator](https://www.niehs.nih.gov/research/resources/software/biostatistics/art/index.cfm). Currently available profiles in ART are shown below. If your sequencer is newer than the available profiles, choose the the most recent model with a similar preparation workflow and read length. For example, training using `HSXn` with a data produced from a NovaSeq 6000.

```
GA1 - GenomeAnalyzer I (36bp,44bp), GA2 - GenomeAnalyzer II (50bp, 75bp)
HS10 - HiSeq 1000 (100bp),          HS20 - HiSeq 2000 (100bp),      HS25 - HiSeq 2500 (125bp, 150bp)
HSXn - HiSeqX PCR free (150bp),     HSXt - HiSeqX TruSeq (150bp),   MinS - MiniSeq TruSeq (50bp)
MSv1 - MiSeq v1 (250bp),            MSv3 - MiSeq v3 (250bp),        NS50 - NextSeq500 v2 (75bp)
```

Preprocessing is multi-threaded. Specifying multiple threads, e.g. `threads=8`, will improve preprocessing performance.

### "End-to-end" example

The `paper` directory includes an `example.sh` script that downloads the HG002 short-read sequencing data and the GIAB SV calls, aligns the reads with BWA, calls SNVs with GATK, and then genotypes those SVs with NPSV using a representative workflow.

Aspects of this script are specific to the local computing infrastructure (e.g., directory paths, number of cores, executable paths) and so will need to be modified prior to use. The script assumes you have a customized version of [Truvari](https://github.com/mlinderm/truvari/tree/genotype_stats) installed.

## SV refining

NPSV-deep includes experimental support for automatically identifying "better" SV representations during genotyping using the simulated data. This workflow is implemented with a "proposal" step that generates possible alternate SV representations and a "refining" step that updates the genotype for the original SV from the alternate SV whose simulated data is most similar to the real data.

### Prerequisites

SV proposal requires a BED file (`--simple-repeats-bed`) derived from the UCSC Genome Browser [simpleRepeats.txt.gz](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/simpleRepeat.txt.gz) table dump that contains the standard BED columns plus the repeat period, number of copies and consensus repeat sequence. Alternative representations will only be generated for variants that overlap regions in this file. For convenience `simple_repeats.b37.bed.gz` and the index file (along with the hg38 version `simple_repeats.hg38.bed.gz`) are available at <http://skylight.middlebury.edu/~mlinderman/data/simple_repeats.b37.bed.gz>. To download these files in the Docker container:
```
curl -k https://www.cs.middlebury.edu/~mlinderman/data/simple_repeats.b37.bed.gz -o /data/simple_repeats.b37.bed.gz
curl -k https://www.cs.middlebury.edu/~mlinderman/data/simple_repeats.b37.bed.gz.tbi -o /data/simple_repeats.b37.bed.gz.tbi 
```
### Workflow

To generate possible alternate representations, use the `propose` sub-command for `npsv2`, e.g.

```plaintext
npsv2 command=propose \
    reference=/data/human_g1k_v37.fasta \
    refine.simple_repeats_path=/data/simple_repeats.b37.bed.gz \
    input=tests/data/1_1865644_1866241_DEL.vcf \
    output=tests/results/1_1865644_1866241_DEL.propose.vcf.gz \
    refine.all_alignments=true
```

The `tests/results/1_1865644_1866241_DEL.propose.vcf.gz` file contains the original SV along with the proposed alternative descriptions (linked by the "INFO/ORIGINAL" field). Since we specified `refine.all_alignments=true`, the proposer will generate all possible start positions for the SV within the repetitive region, ~2200 variants. That can be reduced with read support filtering via the `filter` subcommand. Alternately, not setting `all_alignments` will use small maximum number of proposals based on realigning the putative SV within the repeat.

Then genotype the expanded set of putative variant. Mulitple threads are recommend (see below), since we are genotyping numerous SVs.

```plaintext
npsv2 command=genotype \
    threads=4 \
    reference=/data/human_g1k_v37.fasta \
    input=tests/results/1_1865644_1866241_DEL.propose.vcf.gz \
    reads=tests/data/1_1861644_1871561.bam \
    stats_path=tests/data/stats.json \
    output=tests/results/1_1865644_1866241_DEL.propose.npsv2.vcf.gz \
    load_reference=true
```

Then select the best of the proposed representations with the `refine` sub-command. Refinement will update the original VCF with genotypes for the best representation.

```plaintext
npsv2 command=refine \
	input=tests/results/1_1865644_1866241_DEL.propose.npsv2.vcf.gz \
    output=tests/results/1_1865644_1866241_DEL.propose.npsv2.refine.vcf.gz
```

When reviewing the pileup, the GIAB SV description appears to be "left shifted" from the true location as estimated from long-read sequencing data (approximately 1:1866429-1867023). NPSV-deep (and other tools) incorrectly genotype the original SV description as homozygous reference. The SV refine algorithm selects the alternative description where the actual data is most similar to simulated data for non-reference genotypes. The VCF entries produced by `refine` (shown below for this example) contain the alternate and original genotypes, the alternate and original distances (smaller is better) and the alternate SV description. For this variant, `refine` selects 1:1866441-1867038 as the best SV description. The alternate SV description is correctly genotyped as heterozygous. 

```
GT:DS:DHFFC:FS:SOR:ALTS:OGT:ODS:CL	0/1:0.642001,0.196222,0.844626:0.447:9:2.825:2205:0/0:0.0597,0.8897,0.9322:1866441_1867038
```

Note that due to the random simulations the distances will differ between runs.

## FAQ

### Developing on an Apple M1

An experimental Dockerfile is included for testing on systems with an Apple M1. To build an Apple-specific container:
```
docker build -f Dockerfile.m1 -t npsv2-m1 .
```
and then start the npsv2-m1 container.

### Parallelization

NPSV-deep can perform genotyping and other per-variant operations in parallel (controlled via the `threads` parameter). While inference is typically perfomed on the CPU only, training is typically performed on a CUDA-capable GPU.

### Data availability

The `example.sh` script in `paper` directory includes an example of downloading and preparing both the HG002 short-read sequencing data and the GIAB SV calls for use with the NPSV genotyper. Similar NGS data is available for the parental [HG003](ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG003_NA24149_father/NIST_HiSeq_HG003_Homogeneity-12389378/HG003_HiSeq300x_fastq/140721_D00360_0044_AHA66RADXX) and [HG004](ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data//AshkenazimTrio/HG004_NA24143_mother/NIST_HiSeq_HG004_Homogeneity-14572558/HG004_HiSeq300x_fastq/140818_D00360_0046_AHA5R5ADXX) samples. The NA12878 "Platinum Genomes" NGS data is available in the European Nucleotide Archive under project [PRJEB3381](https://www.ebi.ac.uk/ena/browser/view/PRJEB3381). The Polaris SV call set is available via [GitHub](https://github.com/Illumina/Polaris). The HGSCV2 call set is available via the [IGSR](https://www.internationalgenome.org/data-portal/data-collection/hgsvc2) as is the [SRS data](https://www.internationalgenome.org/data-portal/data-collection/30x-grch38) used for training. The Syndip callset used for model selection is available via [GitHub](https://github.com/lh3/CHM-eval).
