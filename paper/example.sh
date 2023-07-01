#!/usr/bin/env bash

# Run NPSV-deep "end-to-end" on GIAB data (NGS data and SVs). Note that aspects of this script
# are specific to the local computing infrastructure.

#SBATCH --job-name=example
#SBATCH --output=example-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=88G
#SBATCH --exclude node018,node019                              
#SBATCH --partition=long
#SBATCH --time=7-00:00:00

set -euo pipefail

GATK=/storage/mlinderman/ngs/resources/bcbio/1.2.8/anaconda/bin/gatk

REFERENCE=/storage/mlinderman/projects/sv/npsv2-experiments/resources/human_g1k_v37.fasta
MODEL_CACHE_DIR=/storage/mlinderman/projects/sv/npsv2-experiments/.cache
mkdir -p $MODEL_CACHE_DIR

## Setup environment
export TMPDIR=$(mktemp -d --tmpdir="$SCRATCH") || exit 1
trap "rm -rf ${TMPDIR};" 0
THREADS=${SLURM_CPUS_PER_TASK:-1}

## Download NGS data for HG002 (corresponding to approximately 25x coverage)
# Adapted from: https://github.com/brentp/duphold/blob/master/paper/evaluation.sh
ALIGN_FILE=hg002.bam
wget -N -r ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq/140528_D00360_0018_AH8VC6ADXX/

for f in $(find . -type f -name "*_R1_*.fastq.gz" | sort); do zcat $f; done | bgzip -@ 8 -c > hg002_R1.fastq.gz &
for f in $(find . -type f -name "*_R2_*.fastq.gz" | sort); do zcat $f; done | bgzip -@ 8 -c > hg002_R2.fastq.gz &
wait

## Align NGS data using simplified pipeline
bwa mem -c 250 -M -v 1 -t $THREADS \
    -R '@RG\tID:HG002\tSM:HG002\tPL:illumina\tPU:HG002\tLB:HG002' \
    $REFERENCE \
    hg002_R1.fastq.gz hg002_R2.fastq.gz | \
    samblaster -q -M --addMateTags | \
    samtools sort -T "${TMPDIR}/samtools" -@ 4 -m 16G --output-fmt BAM --reference $REFERENCE -o $ALIGN_FILE
samtools index $ALIGN_FILE

# Call SNVs using GATK haplotype caller and phase with WhatsHap
SNV_FILE=hg002.vcf.gz
unset JAVA_HOME && $GATK \
    --java-options "-Xmx16g -Djava.io.tmpdir=$TMPDIR" \
    HaplotypeCaller \
    -R $REFERENCE \
    -I $ALIGN_FILE \
    --output $SNV_FILE

PHASED_SNV_FILE="${SNV_FILE%.vcf.gz}.phased.vcf.gz"
whatshap phase --reference="$REFERENCE" --indels -o "$PHASED_SNV_FILE" "$SNV_FILE" "$ALIGN_FILE"
bcftools index -t "$PHASED_SNV_FILE"

## Download GIAB variants, BED files and filter callset
wget -N ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release//AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/HG002_SVs_Tier1_v0.6.vcf.gz
wget -N ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release//AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/HG002_SVs_Tier1_v0.6.vcf.gz.tbi
wget -N ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release//AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/HG002_SVs_Tier1_v0.6.bed
wget -N ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release//AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/HG002_SVs_Tier1plusTier2_v0.6.1.bed

GIAB_VCF=HG002_SVs_Tier1_v0.6.vcf.gz
GIAB_BED=HG002_SVs_Tier1_v0.6.bed
GIAB_ALL_TIERS_BED=HG002_SVs_Tier1and2_v0.6.bed
GIAB_FILTERED_VCF="${GIAB_VCF%.vcf.gz}.genotyped.passing.tier1and2.vcf.gz"

cat HG002_SVs_Tier1_v0.6.bed HG002_SVs_Tier1plusTier2_v0.6.1.bed | sort -k1,1 -k2,2n > "${TMPDIR}/tmp.bed"
bedtools merge -i "${TMPDIR}/tmp.bed" > $GIAB_ALL_TIERS_BED

# Alternately filter with $GIAB_ALL_TIERS_BED
bcftools view -g ^miss -f 'PASS,LongReadHomRef' -R $GIAB_BED -i '(INFO/sizecat != "20to49")' $GIAB_VCF | \
	bcftools view -Oz -o $GIAB_FILTERED_VCF -e 'FILTER == "LongReadHomRef" && GT != "ref"'
bcftools index -t $GIAB_FILTERED_VCF

## Genotype SVs

# Preprocess alignment file prior to genotyping to obtain coverage, insert size distribution, etc.
npsv2 command=preprocess \
    threads=$THREADS \
    reference=$REFERENCE \
    reads=$ALIGN_FILE \
    output="stats.json"

# Genotype SVs
GENOTYPED_VCF="${GIAB_FILTERED_VCF%.vcf.gz}.npsv2.vcf.gz"
npsv2 command=genotype \
    threads=$THREADS \
    reference=$REFERENCE \
    pileup.snv_vcf_input="$SNV_FILE" \
    input=$GIAB_FILTERED_VCF \
    output=$GENOTYPED_VCF \
    reads=$ALIGN_FILE \
    stats_path="stats.json" \
    load_reference=true \
    cache_dir="$MODEL_CACHE_DIR"

## Concordance analysis with Truvari

GIAB_FILTERED_DEL_VCF="${GIAB_FILTERED_VCF%.vcf.gz}.DEL.vcf.gz"
GIAB_FILTERED_INS_VCF="${GIAB_FILTERED_VCF%.vcf.gz}.INS.vcf.gz"

GENOTYPED_DEL_VCF="${GENOTYPED_VCF}.DEL.vcf.gz"
GENOTYPED_INS_VCF="${GENOTYPED_VCF}.INS.vcf.gz"

filter_vcf() {
    bcftools view -Oz -o "$3" -i "(SVTYPE ~ \"^${1}\")" "$2"
	bcftools index -t "$3"
}

filter_vcf DEL $GIAB_FILTERED_VCF $GIAB_FILTERED_DEL_VCF
filter_vcf INS $GIAB_FILTERED_VCF $GIAB_FILTERED_INS_VCF
filter_vcf DEL $GENOTYPED_VCF $GENOTYPED_DEL_VCF
filter_vcf INS $GENOTYPED_VCF $GENOTYPED_INS_VCF

run_truvari() {
    rm -rf "truvari_${1}"
    truvari bench \
        -f $REFERENCE \
        -b $3 --bSample HG002\
        -c $2 --cSample HG002 \
        -o "truvari_${1}" \
        --includebed $GIAB_BED \
        --sizemax 15000000 -s 50 -S 30 --pctsim=0 -r 20 -O 0.6
}

(
    module load truvari/genotype
    run_truvari DEL $GENOTYPED_DEL_VCF $GIAB_FILTERED_DEL_VCF
    run_truvari INS $GENOTYPED_INS_VCF $GIAB_FILTERED_INS_VCF
)
