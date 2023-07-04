import typing
import pysam
import pysam.bcftools as bcftools
from scipy.special import comb


def index_variant_file(filename: str):
    """Index variant file"""
    # There is a file handle leak in pysam.tabix_index, so we use bcftools index directly
    if filename.endswith("vcf.gz"):
        bcftools.index("-t", filename, catch_stdout=False)
    elif filename.endswith(".bcf"):
        bcftools.index("-c", filename, catch_stdout=False)


def bcftools_format(filename: str) -> str:
    """Return bcftools type string based on filename extensions, e.g. 'vcf.gz' -> 'z'"""
    if filename.endswith("vcf.gz"):
        return "z"
    elif filename.endswith(".bcf"):
        return "b"
    else:
        return "v"


def genotype_field_len(num_alt: int, ploidy: int) -> int:
    """Number of entries in VCF genotype likelihood (or other 'G') field

    Args:
        num_alt (int): Number of alternate alleles
        ploidy (int): Ploidy

    Returns:
        int: Number of entries in genotype field
    """
    return comb(ploidy + num_alt, num_alt, exact=True)


# Adapted from nucleus:
# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L718
def _genotypes_in_genotype_field(num_alt: int, ploidy=2) -> typing.Generator[typing.Tuple[int, ...], None, None]:
    """Generate VCF allele indices (genotype) in order of the VCF genotype likelihood (or other 'G') field

    Args:
        num_alt (int): Number of alternate alleles
        ploidy (int, optional): Ploidy. Defaults to 2.

    Raises:
        NotImplementedError: Specified ploidy is not supported

    Yields:
        Tuple[int,...]: Tuple of genotypes for each index in genotypes field, e.g. (0,0), (0,1)...
    """
    if ploidy == 1:
        for i in range(num_alt + 1):
            yield (i,)
    elif ploidy == 2:
        for j in range(num_alt + 1):
            for i in range(j + 1):
                yield (i, j)
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")


# Adapted from nucleus:
# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L793
def genotype_field_index(allele_indices: typing.Sequence[int]) -> int:
    """Determine index in VCF genotype likelihood (or other 'G') field for genotype

    Args:
        allele_indices (Sequence[int]): Genotype, e.g. (0,1)

    Raises:
        NotImplementedError: Specified ploidy is not supported

    Returns:
        int: Index in genotype field
    """
    if len(allele_indices) == 1:
        return allele_indices[0]
    elif len(allele_indices) == 2:
        a1, a2 = sorted(allele_indices)
        return a1 + (a2 * (a2 + 1) // 2)
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")


# Adapted from nucleus:
# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L820
def allele_indices_from_genotype_field_index(index: int, num_alt: int, ploidy=2) -> typing.Tuple[int, ...]:
    """VCF allele indices (genotype) for index in VCF genotype likelihood (or other 'G') field

    Args:
        index (int): Index in genotype field
        num_alt (int): Number of alternate alleles for variant
        ploidy (int, optional): [description]. Defaults to 2.

    Raises:
        NotImplementedError: Specified ploidy is not supported

    Returns:
        typing.Tuple[int,...]: Genotype, e.g. (0,1)
    """
    if ploidy == 1:
        return (index,)
    elif ploidy == 2:
        # Adapted from nucleus. There is a potentially more efficient algorithm (instead of creating all of the genotypes)
        genotypes = list(_genotypes_in_genotype_field(num_alt, ploidy))
        return genotypes[index]
    else:
        raise NotImplementedError("Only ploidy <= 2 is currently supported")
