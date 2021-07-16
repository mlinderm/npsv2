import pysam
import pysam.bcftools as bcftools

def index_variant_file(filename: str):
    # There is a file handle leak in pysam.tabix_index, so we use bcftools index directly
    # TODO: Detect file format in another way that doesn't generate spurious warnings
    with pysam.VariantFile(filename, "r") as variant_file:
        if variant_file.format == "VCF" and variant_file.compression == "BGZF":
            bcftools.index("-t", filename, catch_stdout=False)
        elif variant_file.format == "BCF":
            bcftools.index("-c", filename, catch_stdout=False)

def bcftools_format(filename: str):
    if filename.endswith("vcf.gz"):
        return "z"
    elif filename.endswith(".bcf"):
        return "b"
    else:
        return "v"
