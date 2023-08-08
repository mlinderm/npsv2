#!/usr/bin/env bash

# Benchmark SV genotypers. Note that aspects of this script are specific to the local computing infrastructure.

# sbatch scripts/benchmarking npsv2 resources/HG002_SVs_Tier1_v0.6.genotyped.passing.tier1and2.b37.DEL.vcf.gz resources/HG002-ready.b37.bam

#SBATCH --job-name=benchmarking
#SBATCH --output=benchmarking-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G                              
#SBATCH --partition=long
#SBATCH --time=7-00:00:00
#SBATCH --exclude=node018,node019

set -euo pipefail

# Metadata
THREADS=${SLURM_CPUS_PER_TASK:-1}
TIME_FORMAT="%E,%M,%U,%S,${THREADS},${SLURM_JOB_ID:-"NA"}"
OUTPUT_DIR=

# Data paths
NPSV2_EXP=/storage/mlinderman/projects/sv/npsv2-experiments
RESOURCES="${NPSV2_EXP}/resources"
REFERENCE="${RESOURCES}/human_g1k_v37.fasta"
SNV_FILE=/storage/mlinderman/ngs/resources/ashkenazi-trio/b37/final/2021-10-14_project/ashkenazi-gatk-haplotype-annotated.vcf.gz

usage()
{
    cat << EOF
usage: $(basename "$0") [options] GENOTYPER VCF BAM
Benchmark SV genotypers
Options:
  -h            Print this message
  -o            Output directory, defaults to temporary directory
  -r            Path to reference fasta, defaults to $REFERENCE
  -s            Path to SNV VCF file, defaults to $SNV_FILE
EOF
}

while getopts "ho:r:b:s:" Option
do
    case $Option in
        o)
            OUTPUT_DIR=$OPTARG
            ;;
        r)
            REFERENCE="$(realpath --no-symlinks "$OPTARG")"
            ;;
        s)
            SNV_FILE="$(realpath --no-symlinks "$OPTARG")"
            ;;
        h)
            usage
            exit 0
            ;;
        ?)
            usage
            exit 85
            ;;
    esac
done

shift $((OPTIND-1))
if [[ $# -ne 3 ]]; then
    >&2 echo "Error: Missing positional arguments"
    >&2 usage
    exit 1
fi

GENOTYPER=$1
VCF_FILE=$(realpath --no-symlinks "$2")
BAM_FILE=$(realpath --no-symlinks "$3")
MODE="NA"

READ_EXT=".${BAM_FILE##*.}"
module load samtools
SAMPLE=$(samtools view -H $BAM_FILE | grep '^@RG' | sed 's/.*SM:\(\w\+\).*/\1/' | sort -u)

# Setup environment
export TMPDIR=$(mktemp -d --tmpdir="$SCRATCH") || exit 1
trap "rm -rf ${TMPDIR};" 0

if [[ -z $OUTPUT_DIR ]]; then
    OUTPUT_DIR="${TMPDIR}/output"
fi
mkdir -p $OUTPUT_DIR

if [[ $GENOTYPER == "npsv2_pre" ]]; then
    STATS_PATH="$(basename --suffix=$READ_EXT $BAM_FILE).npsv2.stats.json"
    
    # Run preprocessing
    cd $OUTPUT_DIR && \
    /bin/time -o "${STATS_PATH}.time" -f "Timing,npsv2,${MODE},preprocessing,${TIME_FORMAT}" \
    npsv2 command=preprocess \
       threads=$THREADS \
        reference=$REFERENCE \
        reads=$BAM_FILE \
        output="$STATS_PATH"

elif [[ $GENOTYPER == "npsv2" ]]; then
    MODE="default"
    GENOTYPED_VCF="$(basename "${VCF_FILE%.vcf.gz}.npsv2.vcf.gz")"
    
    cd $OUTPUT_DIR && \
    /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,npsv2_all,${MODE},genotyping,${TIME_FORMAT}" \
    npsv2 command=genotype \
        threads=$THREADS \
        reference=$REFERENCE \
        pileup.snv_vcf_input="$SNV_FILE" \
        input=$VCF_FILE \
        output=$GENOTYPED_VCF \
        reads=$BAM_FILE \
        stats_path="${RESOURCES}/$(basename -s ".bam" $BAM_FILE).stats.json" \
		load_reference=true \
        cache_dir="${NPSV2_EXP}/.cache"

elif [[ $GENOTYPER == "npsv_pre" ]]; then
    module load npsv/master

    MODE="default"

    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        GENOME="${NPSV_ROOT}/etc/human_g1k_v37.genome"
    elif [[ "$(basename $BAM_FILE)" == "NA12878-ready.hg38.bam" ]]; then 
        GENOME="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.genome"
    elif [[ "$(basename $BAM_FILE)" == "NA12878.final.cram" ]]; then
        GENOME="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.genome"
    else
        >&2 echo "Error: Unknown BAM file"
        exit 1
    fi

    STATS_PATH="${OUTPUT_DIR}/$(basename --suffix=$READ_EXT $BAM_FILE).npsv.stats.json"
    
    /bin/time -o "${STATS_PATH}.time" -f "Timing,npsv,${MODE},preprocessing,${TIME_FORMAT}" \
    npsvg preprocess \
        -r $REFERENCE \
        -b $BAM_FILE \
        --genome "$GENOME" \
        -o "$STATS_PATH"
    
elif [[ $GENOTYPER == "npsv" ]]; then
    # NPSV requires END tags and vector SVLEN
    NPSV_INPUT_VCF="${TMPDIR}/npsv.vcf.gz"
    bcftools +fill-tags -Ov $VCF_FILE -- -t END | \
        sed 's/^##INFO=<ID=SVLEN,Number=1/##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference between REF and ALT alleles">/' | \
        bgzip > $NPSV_INPUT_VCF
    bcftools index -t $NPSV_INPUT_VCF
    
    module load npsv/master

    MODE="default"
    NPSV_PREFIX="$(basename -s ".vcf.gz" $VCF_FILE)"

    FILTER_BED=""
    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        GENOME="${NPSV_ROOT}/etc/human_g1k_v37.genome"
        GAPS="${NPSV_ROOT}/etc/human_g1k_v37.gaps.bed.gz"
        PROFILE="HS25"
        FILTER_BED="--filter-bed ${RESOURCES}/HG002_SVs_Tier1_v0.6.bed"
    elif [[ "$(basename $BAM_FILE)" == "NA12878-ready.hg38.bam" ]]; then 
        GENOME="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.genome"
        GAPS="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.gaps.bed.gz"
        PROFILE="HS25"
    elif [[ "$(basename $BAM_FILE)" == "NA12878.final.cram" ]]; then
        GENOME="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.genome"
        GAPS="${NPSV_ROOT}/etc/Homo_sapiens_assembly38.gaps.bed.gz"
        PROFILE="HSXn"
    else
        >&2 echo "Error: Unknown BAM file"
        exit 1
    fi

    # Load reference genome into shared memory
    bwa shm $REFERENCE
    trap "rm -rf ${TMPDIR}; bwa shm -d;" 0

    GENOTYPED_VCF="${OUTPUT_DIR}/${NPSV_PREFIX}.npsv.vcf"

    /bin/time -o "${GENOTYPED_VCF}.gz.time" -f "Timing,npsv,${MODE},genotyping,${TIME_FORMAT}" \
    npsv -v \
        --tempdir $TMPDIR \
        --threads $THREADS \
        -r $REFERENCE \
        --genome $GENOME \
        --gaps $GAPS \
        --stats-path "${RESOURCES}/$(basename "${BAM_FILE%.*}.stats.json")" \
        --profile $PROFILE \
        $FILTER_BED \
        -i $NPSV_INPUT_VCF \
        -b $BAM_FILE \
        -o $OUTPUT_DIR \
        --prefix $NPSV_PREFIX
    
    
    bgzip -f $GENOTYPED_VCF && tabix "${GENOTYPED_VCF}.gz"
    GENOTYPED_VCF="${GENOTYPED_VCF}.gz"    

elif [[ $GENOTYPER == "svviz2_mapq" ]]; then
    MODE="mapq"
    REPORT_DIR="${OUTPUT_DIR}/$(basename -s ".vcf.gz" $VCF_FILE).svviz2_reports"
    mkdir -p $REPORT_DIR

    SVVIZ2_INPUT_VCF="${TMPDIR}/svviz2.vcf.gz"
    bcftools annotate -Oz -o $SVVIZ2_INPUT_VCF --set-id +'%CHROM\_%POS\_%END' $VCF_FILE
    bcftools index -t $SVVIZ2_INPUT_VCF

    # To complete svviz2 within cluster limits we need to parallelize
    VCF_COUNT=$(bcftools index --nrecords $SVVIZ2_INPUT_VCF)
    SPAN_FILE="${TMPDIR}/svviz2_spans.txt"

    python3 - <<END
span=($VCF_COUNT + $THREADS - 1) // $THREADS
with open("$SPAN_FILE", "w") as spans:
    for i in range(0,$VCF_COUNT,span):
        print(i, min(i + span -1, $VCF_COUNT - 1), sep="\t", file=spans)
END

    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename "${VCF_FILE%.vcf.gz}.svviz2_mapq.vcf.gz")"
    
    # This assumes REF_CACHE has been constructed to support CRAM files 
    (
        module load svviz2
        REF_CACHE="${REFERENCE%.fasta}_cache/%2s/%2s/%s" \
        /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,svviz2,${MODE},genotyping,${TIME_FORMAT}" \
        parallel -j $THREADS --colsep '\t' \
        svviz2 \
            --report-only \
            --outdir $REPORT_DIR \
            --first-variant {1} --last-variant {2} \
            --ref $REFERENCE \
            --variants $SVVIZ2_INPUT_VCF \
            $BAM_FILE \
        :::: $SPAN_FILE
    )

    /bin/time -a -o "${GENOTYPED_VCF}.time" -f "Timing,svviz2,${MODE},postprocessing,${TIME_FORMAT}" \
    ${NPSV2_EXP}/scripts/svviz2vcf \
        --model mapq \
        -i $VCF_FILE \
        -r $REPORT_DIR \
		-s $(basename -s $READ_EXT $BAM_FILE | sed 's/\./_/g') \
		-o /dev/stdout | \
		bcftools reheader -s <(echo -e $SAMPLE) | \
		sed 's/\bnan\b/./g' | \
		bgzip > $GENOTYPED_VCF
	bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "svtyper" ]]; then
    # svtyper requires END tag in VCF
    module load svtyper/0.7.1
    
    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename -s ".vcf.gz" $VCF_FILE).svtyper.vcf.gz"
    /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,svtyper,${MODE},genotyping,${TIME_FORMAT}" \
    svtyper \
        -i <(bcftools +fill-tags $VCF_FILE -- -t END | bcftools view -G) \
        -B $BAM_FILE | \
		sed 's/=""/="/' | \
		bgzip -c > $GENOTYPED_VCF
	bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "paragraph" ]]; then   
    PARAGRAPH_INPUT_VCF="${TMPDIR}/paragraph.vcf.gz"
    ${NPSV2_EXP}/scripts/fixSVVCF --pad_alleles -r $REFERENCE -i $VCF_FILE | \
        bcftools sort -Oz -o $PARAGRAPH_INPUT_VCF
    bcftools index -t $PARAGRAPH_INPUT_VCF

    module load paragraph/2.4a
    
    PARAGRAPH_SAMPLE_FILE="${TMPDIR}/paragraph.sample.txt"
    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        cat <<EOF > $PARAGRAPH_SAMPLE_FILE
id	path	depth	read length	sex
HG002	$BAM_FILE	25	148	male
EOF
    elif [[ "$(basename $BAM_FILE)" == "NA12878-ready.hg38.bam" ]]; then
        cat <<EOF > $PARAGRAPH_SAMPLE_FILE
id	path	depth	read length	sex
NA12878	$BAM_FILE	49	101	female
EOF
    elif [[ "$(basename $BAM_FILE)" == "NA12878.final.cram" ]]; then
        cat <<EOF > $PARAGRAPH_SAMPLE_FILE
id	path	depth	read length	sex
NA12878	$BAM_FILE	31	150	female
EOF
    else
        >&2 echo "Error: Unknown BAM file"
        exit 1
    fi

    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename "${VCF_FILE%.vcf.gz}.paragraph.vcf.gz")"

	/bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,paragraph,${MODE},genotyping,${TIME_FORMAT}" \
    multigrmpy.py \
		--scratch-dir $TMPDIR -t $SLURM_CPUS_PER_TASK \
		-i $PARAGRAPH_INPUT_VCF \
		-m $PARAGRAPH_SAMPLE_FILE \
		-r $REFERENCE \
		-o $TMPDIR
	bcftools index -t "${TMPDIR}/genotypes.vcf.gz"   
    
    # Need to set ploidy to 2 for truvari comparison with GIAB
    bcftools +fixploidy -Oz -o $GENOTYPED_VCF "${TMPDIR}/genotypes.vcf.gz" -- -f 2    
	bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "graphtyper" ]]; then
    module load graphtyper/2.5.1

    GRAPHTYPER_BAMS="${TMPDIR}/graphtyper.bams.list"
    echo "$BAM_FILE" > $GRAPHTYPER_BAMS

    REGION_FILE="${TMPDIR}/region.list"
    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        awk '{ print $1 ":" $2+1 "-" $3; }' "${RESOURCES}/HG002_SVs_Tier1and2_v0.6.bed" > $REGION_FILE
    elif [[ "$(basename $BAM_FILE)" == "NA12878-ready.hg38.bam" || "$(basename $BAM_FILE)" == "NA12878.final.cram" ]]; then
       awk '$2 ~ /^SN:(chr)?[[:digit:]XY]+$/ { print substr($2,4) ":1-" substr($3,4); }' "${REFERENCE%.fasta}.dict" > $REGION_FILE
    else
        >&2 echo "Error: Unknown BAM file"
        exit 1
    fi
    
    GRAPHTYPER_OUTPUT_DIR="${TMPDIR}/graphtyper"
    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename "${VCF_FILE%.vcf.gz}.graphtyper.vcf.gz")"

    /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,graphtyper,${MODE},genotyping,${TIME_FORMAT}" \
    graphtyper \
        genotype_sv $REFERENCE $VCF_FILE \
		--output=$GRAPHTYPER_OUTPUT_DIR \
		--sams=$GRAPHTYPER_BAMS \
		--region_file=$REGION_FILE \
		--force_no_copy_reference \
        --threads=1
	
    bcftools concat --naive $GRAPHTYPER_OUTPUT_DIR/**/*.vcf.gz | \
        bcftools view -i '((SVTYPE ~ "^DEL" || SVTYPE ~ "^INS" || SVTYPE ~ "^DUP") && INFO/SVMODEL=="AGGREGATED")' | \
        sed '/^#/!s/DUP/INS/g' | \
        bcftools sort -Oz -o $GENOTYPED_VCF -
	bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "delly" ]]; then
    module load delly/0.8.3

    DELLY_INPUT_BCF="${TMPDIR}/delly.bcf"
    bcftools view -Ob -o $DELLY_INPUT_BCF $VCF_FILE
    bcftools index -c $DELLY_INPUT_BCF

    GENOTYPED_BCF="${OUTPUT_DIR}/$(basename "${VCF_FILE%.vcf.gz}.delly.bcf")"
    GENOTYPED_VCF="${GENOTYPED_BCF%.bcf}.vcf.gz"
    /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,delly,${MODE},genotyping,${TIME_FORMAT}" \
    delly call \
        -g $REFERENCE \
        -x "${DELLY_HOME}/excludeTemplates/human.hg19.excl.tsv" \
        -v $DELLY_INPUT_BCF \
        -o $GENOTYPED_BCF \
        $BAM_FILE
 
    bcftools view -Oz -o $GENOTYPED_VCF $GENOTYPED_BCF
    bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "sv2" ]]; then
    module load sv2/1.5

    if [[ "$(basename $REFERENCE)" == "human_g1k_v37.fasta" ]]; then
        SV2_REF_ARG="hg19"
    else
        SV2_REF_ARG="hg38"
    fi

    SV2_PED_FILE="${TMPDIR}/sv2.ped"
    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        cat <<EOF > $SV2_PED_FILE
8392	HG002	HG003	HG004	1	0       
EOF
    else
        cat <<EOF > $SV2_PED_FILE
1491	NA12878	NA12891	NA12892	2	0       
EOF
    fi

    # SV2 requires END tags
    SV2_INPUT="${TMPDIR}/sv2_input.vcf.gz"
    bcftools +fill-tags -Oz -o $SV2_INPUT $VCF_FILE -- -t END

    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename "${VCF_FILE%.vcf.gz}.sv2.vcf.gz")"
	/bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,sv2,${MODE},genotyping,${TIME_FORMAT}" sv2 \
        -M \
        -g $SV2_REF_ARG \
        -i $BAM_FILE \
        -v $SV2_INPUT \
        -snv $SNV_FILE \
        -p $SV2_PED_FILE \
        -O $TMPDIR \
        -o $(basename -s ".vcf.gz" $GENOTYPED_VCF)    

    SV2_OUTPUT="${TMPDIR}/sv2_genotypes/$(basename -s ".vcf.gz" $GENOTYPED_VCF).vcf"

    sed '/^#/b; s/NA/./g' $SV2_OUTPUT | \
        sed 's/\(^##FILTER=<ID=GAP\)>/\1,Description="Missing description">/' | \
        bcftools +fixploidy -- -f 2 - | \
        bcftools norm -f $REFERENCE -c s - | \
        bcftools annotate -x INFO/GENES -Oz -o "${GENOTYPED_VCF}" -
	
    bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "genomestrip" ]]; then
    module load picard
    module load GenomeSTRiP/2.00.1958 

    if [[ "$(basename $REFERENCE)" == "human_g1k_v37.fasta" ]]; then
        SV_METADATA_DIR=/storage/mlinderman/ngs/resources/genomestrip/1000G_phase1
        SV_METADATA_PREFIX="human_g1k_v37"
    else
        SV_METADATA_DIR=/storage/mlinderman/ngs/resources/genomestrip/Homo_sapiens_assembly38
        SV_METADATA_PREFIX="Homo_sapiens_assembly38"
    fi

    GENOMESTRIP_DIR="${OUTPUT_DIR}/$(basename -s ".vcf.gz" $VCF_FILE).genomestrip"
    SAMPLE_METADATA_DIR="${GENOMESTRIP_DIR}/genomestrip_preprocessing"
    mkdir -p $SAMPLE_METADATA_DIR

    GENOMESTRIP_INPUT="${TMPDIR}/genomestrip-input.vcf.gz"
    bcftools sort -Oz -o $GENOMESTRIP_INPUT $VCF_FILE
    bcftools index -t $GENOMESTRIP_INPUT

    GENOTYPED_VCF="${OUTPUT_DIR}/$(basename -s ".vcf.gz" $VCF_FILE).genomestrip.vcf.gz"

    /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,genomestrip,${MODE},preprocessing,${TIME_FORMAT}" \
    java -Xmx4g -cp $CLASSPATH \
		-Djava.io.tmpdir=$TMPDIR \
		org.broadinstitute.gatk.queue.QCommandLine \
		-S $SV_DIR/qscript/SVPreprocess.q \
		-S $SV_DIR/qscript/SVQScript.q \
		-cp $CLASSPATH \
		-gatk $SV_DIR/lib/gatk/GenomeAnalysisTK.jar \
		-configFile $SV_DIR/conf/genstrip_parameters.txt \
		-ploidyMapFile "${SV_METADATA_DIR}/${SV_METADATA_PREFIX}.ploidymap.txt" \
		-R $REFERENCE \
		-I $BAM_FILE \
		-md $SAMPLE_METADATA_DIR \
		-bamFilesAreDisjoint true \
		-jobLogDir "${GENOMESTRIP_DIR}/genomestrip_log" \
		-jobRunner ParallelShell -maxConcurrentRun $SLURM_CPUS_PER_TASK \
		-run

    GENOMESTRIP_GENDER_MAP="${TMPDIR}/genomestrip.gender.txt"
    if [[ "$(basename $BAM_FILE)" == "HG002-ready.b37.bam" ]]; then
        echo -e "HG002\tMale" > $GENOMESTRIP_GENDER_MAP
    elif [[ "$(basename $BAM_FILE)" == "NA12878-ready.hg38.bam" || "$(basename $BAM_FILE)" == "NA12878.final.cram" ]]; then
       echo -e "NA12878\tFemale" > $GENOMESTRIP_GENDER_MAP
    else
        >&2 echo "Error: Unknown BAM file"
        exit 1
    fi

    GENOMESTRIP_OUTPUT="${TMPDIR}/genomestrip.vcf.gz"
    /bin/time -a -o "${GENOTYPED_VCF}.time" -f "Timing,genomestrip,${MODE},genotyping,${TIME_FORMAT}" \
    java -Xmx4g -cp $CLASSPATH \
		-Djava.io.tmpdir=$TMPDIR \
    	org.broadinstitute.gatk.queue.QCommandLine \
    	-S $SV_DIR/qscript/SVGenotyper.q \
		-S $SV_DIR/qscript/SVQScript.q \
		-cp $CLASSPATH \
		-gatk $SV_DIR/lib/gatk/GenomeAnalysisTK.jar \
		-configFile $SV_DIR/conf/genstrip_parameters.txt \
		-rmd $SV_METADATA_DIR \
		-R $REFERENCE \
		-genomeMaskFile "${SV_METADATA_DIR}/${SV_METADATA_PREFIX}.svmask.fasta" \
		-genderMapFile $GENOMESTRIP_GENDER_MAP \
		-ploidyMapFile "${SV_METADATA_DIR}/${SV_METADATA_PREFIX}.ploidymap.txt" \
		-md $SAMPLE_METADATA_DIR \
		-runDirectory $GENOMESTRIP_DIR \
		-vcf $GENOMESTRIP_INPUT \
		-I $BAM_FILE \
		-O $GENOMESTRIP_OUTPUT \
		-bamFilesAreDisjoint true \
		-jobLogDir "${GENOMESTRIP_DIR}/genomestrip_log" \
		-jobRunner ParallelShell -maxConcurrentRun $SLURM_CPUS_PER_TASK \
		-run

    # Remove FORMAT field that seems to be causing bcftools parsing errors
    gzip -dc $GENOMESTRIP_OUTPUT | \
        sed 's/:NA:/:.:/g' | \
        bcftools annotate -Oz -o $GENOTYPED_VCF -x FORMAT/CNF -
    bcftools index -t $GENOTYPED_VCF

elif [[ $GENOTYPER == "samplotml" ]]; then
    GENOTYPED_VCF="$(realpath --no-symlinks ${OUTPUT_DIR})/$(basename "${VCF_FILE%.vcf.gz}.samplotml.vcf.gz")"
    
    WORKDIR="$(realpath --no-symlinks ${OUTPUT_DIR})/$(basename -s ".vcf.gz" $VCF_FILE).samplotml"
    mkdir -p $WORKDIR

    SAMPLOTML_INPUT_VCF="${WORKDIR}/samplot.vcf.gz"
    bcftools +fill-tags -Oz -o $SAMPLOTML_INPUT_VCF $VCF_FILE -- -t END
    bcftools index -t $SAMPLOTML_INPUT_VCF

    CONFIG_FILE="${WORKDIR}/config.yaml"

    cat <<EOF > $CONFIG_FILE
samples:
    ${SAMPLE}: "$BAM_FILE"
fasta:
    data_source: "local" 
    file: "$REFERENCE"
fai:
    data_source: "local"
    file: "${REFERENCE}.fai"
vcf:
    data_source: "local"
    file: "$SAMPLOTML_INPUT_VCF"

image_filename_delimiter: "-"
outdir: "$WORKDIR"
EOF

    (
        # Workaround PS1 unbound error with conda: https://github.com/conda/conda/issues/8186
        set +eu
        module load samplot-ml/0.2
        set -eu

        cd "${SAMPLOTML_HOME}/workflows"

        # Note we had to make some tweaks to the environment and configuration (delete existing samples) to successfully run
        # the tool locally
        /bin/time -o "${GENOTYPED_VCF}.time" -f "Timing,samplotml,${MODE},genotyping,${TIME_FORMAT}" \
        snakemake \
            --configfile $CONFIG_FILE \
            -s samplot-ml-predict.smk \
            -j $THREADS \
            --use-conda \
            --conda-frontend mamba \
            --forcerun
    )

    mv "${WORKDIR}/samplot-ml-results/${SAMPLE}-samplot-ml.vcf.gz" $GENOTYPED_VCF
    bcftools index -t $GENOTYPED_VCF
fi  