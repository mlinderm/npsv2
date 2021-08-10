import pysam
import pysam.bcftools as bcftools
from scipy.special import comb


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


def genotype_field_len(num_alt: int, ploidy: int):
    return comb(ploidy + num_alt, num_alt, exact=True)


# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L718
def _genotypes_in_genotype_field(num_alt: int, ploidy=2):
    if ploidy == 1:
        for i in range(num_alt + 1):
            yield (i,)
    elif ploidy == 2:
        for j in range(num_alt + 1):
            for i in range(j + 1):
                yield (i, j)
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")


# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L793
def genotype_field_index(allele_indices):
    if len(allele_indices) == 1:
        return allele_indices[0]
    elif len(allele_indices) == 2:
        a1, a2 = sorted(allele_indices)
        return a1 + (a2 * (a2 + 1) // 2)
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")


# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L820
def allele_indices_from_genotype_field_index(index, num_alt, ploidy=2):
    if ploidy == 1:
        return index
    elif ploidy == 2:
        # Adapted from nucleus. There is a potentially more efficient algorithm (instead of creating all of the genotypes)
        genotypes = list(_genotypes_in_genotype_field(num_alt, ploidy))
        return genotypes[index]
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")
