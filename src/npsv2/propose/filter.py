import pysam
from tqdm import tqdm
import subprocess
from ..variant import Variant, _reference_sequence
from ..range import Range

from . import ORIGINAL_KEY
import matplotlib.pyplot as plt
import math

def filter_vcf(vcf_path: str, output: str, reference_fasta: str, progress_bar=True):

    k = 30
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        # Create header for destination file
        src_header = src_vcf_file.header
        #assert ORIGINAL_KEY not in src_header.info, f"{ORIGINAL_KEY} already presented in VCF INFO field"
        dst_header = src_header.copy()
        dst_header.add_line(
            f'##INFO=<ID={ORIGINAL_KEY},Number=.,Type=String,Description="Filtered Variants based on presence of unique kmers in sequence or absent of unique kmers">'
        )

        with pysam.VariantFile(output, mode="w", header=dst_header) as dst_vcf_file:
            #src_vcf_file.subset_samples(["HG002"])
            for i, record in enumerate(tqdm(src_vcf_file, desc="Generating proposed SV representations", disable=not progress_bar)):
                variant = Variant.from_pysam(record)

                left_flank = variant.left_flank_region(k)
                right_flank = variant.right_flank_region(k)
                left_ref = _reference_sequence(reference_fasta, left_flank)
                right_ref = _reference_sequence(reference_fasta, right_flank)            
                kmer_all = left_ref + right_ref

                # string of the list of kmers
                jellyfish_input = ""
                for i in range(k):
                    jellyfish_input += kmer_all[i:(k+1)+i] + " "

                jellyfish_command = f"jellyfish query /storage/phansen/npsv2/mer_counts.jf {jellyfish_input}"
                kmerstats = subprocess.check_output(jellyfish_command, shell=True, universal_newlines=True)

                res_split = kmerstats.split("\n")
                min_count = math.inf
                kmer_list = []
                for res in res_split:
                    if res != "":
                        count = int(res.split()[1])
                        if count == 0:
                            # add unique kmer
                            kmer_list.append(res.split()[0])
            
                # query unique kmer in sequencing data
                query_input = ""
                for i in range(len(kmer_list)):
                    query_input += kmer_list[i] + " "
                
                sequence_command = f"jellyfish query /storage/phansen/npsv2/HG002-ready.b37.jf {query_input}"
                querystat = subprocess.check_output(sequence_command, shell=True, universal_newlines=True)

                query_split = querystat.split("\n")
                present = []
                for c in query_split:
                    if c != "":
                        count = int(c.split()[1])
                        if count != 0:
                            # add unique kmer
                            present.append(c.split()[0])

                if len(present) != 0 or len(kmer_list) == 0 or not record.info.get(ORIGINAL_KEY, False):
                    dst_vcf_file.write(record)                    
    dst_vcf_file.close()
