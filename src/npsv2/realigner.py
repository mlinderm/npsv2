from enum import Enum
import pysam
from ._realigner import *
from .pileup import Fragment

def _quality_string(read: pysam.AlignedSegment) -> str:
    return "".join([chr(c) for c in read.query_qualities])    

class AlleleAssignment(Enum):
    AMB = -1
    REF = 0
    ALT = 1

def realign_fragment(realigner: FragmentRealigner, fragment: Fragment, assign_delta=1):
    name = fragment.query_name
    read1_seq = fragment.read1.query_sequence
    read1_qual = _quality_string(fragment.read1)
            
    kw = dict(offset=0) # Conversion already performed by pySAM
    if fragment.read2:
        kw["read2_seq"] = fragment.read2.query_sequence
        kw["read2_qual"] = _quality_string(fragment.read2)
    
    scores = realigner.realign_read_pair(name, read1_seq, read1_qual, **kw)
    
    alt_quality = scores["max_alt_quality"]
    ref_quality = scores["ref_quality"]
    if abs(alt_quality - ref_quality) < assign_delta:
        allele = AlleleAssignment.AMB  
    elif alt_quality > ref_quality: 
        allele = AlleleAssignment.ALT
    else:
        allele = AlleleAssignment.REF

    return allele, ref_quality, alt_quality