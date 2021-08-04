import pysam
from ._native import FragmentRealigner, test_score_alignment, test_realign_read_pair
from .pileup import Fragment, AlleleAssignment, AlleleRealignment

def _quality_string(read: pysam.AlignedSegment) -> str:
    return "".join([chr(c) for c in read.query_qualities])    


def realign_fragment(realigner: FragmentRealigner, fragment: Fragment, assign_delta=1):
    name = fragment.query_name
    read1_seq = fragment.read1.query_sequence
    read1_qual = _quality_string(fragment.read1)
            
    kw = dict(offset=0) # Conversion already performed by pySAM
    if fragment.read2:
        kw["read2_seq"] = fragment.read2.query_sequence
        kw["read2_qual"] = _quality_string(fragment.read2)
    
    ref_quality, ref_break, ref_score, ref_max_score, alt_quality, alt_break, alt_score, alt_max_score = realigner.realign_read_pair(name, read1_seq, read1_qual, **kw)
       
    if alt_quality > (ref_quality + assign_delta): 
        return AlleleRealignment(ref_quality, alt_quality, AlleleAssignment.ALT)
    elif ref_quality > (alt_quality + assign_delta):
        return AlleleRealignment(ref_quality, alt_quality, AlleleAssignment.REF)
    else:
        return AlleleRealignment(ref_quality, alt_quality, AlleleAssignment.AMB)
