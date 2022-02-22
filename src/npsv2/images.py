import datetime, functools, itertools, math, logging, os, random, shutil, subprocess, sys, tempfile, typing
from dataclasses import dataclass, field
import numpy as np
import pysam
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import ray
import hydra
from omegaconf import OmegaConf

from .variant import Variant, _reference_sequence
from .range import Range
from .pileup import Pileup, FragmentTracker, AlleleAssignment, BaseAlignment, Strand, ReadPileup, FragmentPileup, read_start
from . import npsv2_pb2
from .realigner import FragmentRealigner, realign_fragment, AlleleRealignment
from .simulation import RandomVariants, simulate_variant_sequencing, augment_samples
from .sample import Sample


MAX_PIXEL_VALUE = 254.0  # Adapted from DeepVariant

ALIGNED_CHANNEL = 0
REF_PAIRED_CHANNEL = 1
ALT_PAIRED_CHANNEL = 2
ALLELE_CHANNEL = 3
MAPQ_CHANNEL = 4
STRAND_CHANNEL = 5
BASEQ_CHANNEL = 6

REF_ALLELE_CHANNEL = 3
ALT_ALLELE_CHANNEL = 7


def _fragment_zscore(sample: Sample, fragment_length: int, fragment_delta=0):
    return (fragment_length + fragment_delta - sample.mean_insert_size) / sample.std_insert_size


def _realigner(variant, sample: Sample, reference, flank=1000, snv_vcf_path: str=None, alleles: typing.AbstractSet[int]={1}):
    with tempfile.TemporaryDirectory() as dir:
        # Generate index fasta with contigs filtered by alleles. The fasta should include the reference sequence and the 
        # sequence of the alternate alleles specified in `alleles`
        fasta_alleles = sorted({0}.union(alleles))
        assert len(fasta_alleles) >= 2
        fasta_path, ref_contig, alt_contig = variant.synth_fasta(reference_fasta=reference, alleles=fasta_alleles, dir=dir, flank=flank, index_mode=True)

        addl_args = {}
        if snv_vcf_path:
            iupac_fasta_path, *_ = variant.synth_fasta(reference_fasta=reference, alleles=fasta_alleles, dir=dir, flank=flank, ref_contig=ref_contig, alt_contig=alt_contig, snv_vcf_path=snv_vcf_path, index_mode=True)
            addl_args["iupac_fasta_path"] = iupac_fasta_path
        
        # Convert breakpoints to list of tuples (of strings) to be passed into the C++ side
        ref_breakpoints = variant.ref_breakpoints(flank, contig=ref_contig)
        if variant.num_alt == 1:
            breakpoints = [ref_breakpoints + variant.alt_breakpoints(flank, contig=alt_contig)]
        else:
            # Convert index to VCF allele index
            breakpoints = [ref_breakpoints + variant.alt_breakpoints(flank, allele=i, contig=alt_contig[i-1]) for i in sorted(alleles)]
        breakpoints = [tuple(map(lambda x: str(x) if x else "", bps)) for bps in breakpoints]

        return FragmentRealigner(fasta_path, breakpoints, sample.mean_insert_size, sample.std_insert_size, **addl_args)


def _fetch_reads(read_path: str, fetch_region: Range, reference: str = None) -> FragmentTracker:
    fragments = FragmentTracker()
    with pysam.AlignmentFile(read_path, reference_filename=reference) as alignment_file:
        for read in alignment_file.fetch(
            contig=fetch_region.contig, start=fetch_region.start, stop=fetch_region.end
        ):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            fragments.add_read(read)
    return fragments


class ImageGenerator:
    def __init__(self, cfg, num_channels):
        self._cfg = cfg
        self._image_shape = (self._cfg.pileup.image_height, self._cfg.pileup.image_width, num_channels)

        # Helper dictionaries to map to pixel values
        self._aligned_to_pixel = {
            BaseAlignment.ALIGNED: self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MATCH: self._cfg.pileup.match_base_pixel if self._cfg.pileup.render_snv else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MISMATCH: self._cfg.pileup.mismatch_base_pixel if self._cfg.pileup.render_snv else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.SOFT_CLIP: self._cfg.pileup.soft_clip_base_pixel,
            BaseAlignment.INSERT: self._cfg.pileup.insert_base_pixel
        }

        self._allele_to_pixel = {
            AlleleAssignment.AMB: self._cfg.pileup.amb_allele_pixel,
            AlleleAssignment.REF: self._cfg.pileup.ref_allele_pixel,
            AlleleAssignment.ALT: self._cfg.pileup.alt_allele_pixel,
            None: 0,
        }

        self._strand_to_pixel = {
            Strand.POSITIVE: self._cfg.pileup.positive_strand_pixel,
            Strand.NEGATIVE: self._cfg.pileup.negative_strand_pixel,
            None: 0.
        }

    @property
    def image_shape(self):
        return self._image_shape


    def _align_pixel(self, align):
        if isinstance(align, BaseAlignment):
            return self._aligned_to_pixel[align]
        else:
            return [self._aligned_to_pixel[a] for a in align]

    def _zscore_pixel(self, zscore):
        if zscore is None:
            return 0
        else:
            return np.clip(
                self._cfg.pileup.insert_size_mean_pixel + zscore * self._cfg.pileup.insert_size_sd_pixel, 1, MAX_PIXEL_VALUE
            )

    def _allele_pixel(self, realignment: AlleleRealignment):
        if self._cfg.pileup.binary_allele:
            return self._allele_to_pixel[realignment.allele]
        elif realignment.ref_quality is None or math.isnan(realignment.ref_quality) or realignment.alt_quality is None or math.isnan(realignment.alt_quality):
            return 0
        else:
            return np.clip((realignment.alt_quality - realignment.ref_quality) / self._cfg.pileup.max_alleleq * self._cfg.pileup.allele_pixel_range + self._cfg.pileup.amb_allele_pixel, 1, MAX_PIXEL_VALUE)
            
    def _strand_pixel(self, read: pysam.AlignedSegment):
        return self._strand_to_pixel[Strand.NEGATIVE if read.is_reverse else Strand.POSITIVE]

    def _qual_pixel(self, qual, max_qual: int):
        if qual is None:
            return 0
        else:
            return np.minimum(np.array(qual) / max_qual, 1.0) * MAX_PIXEL_VALUE

    def _mapq_pixel(self, qual):
        if qual is None:
            return 0
        elif self._cfg.pileup.discrete_mapq:
            if qual == 0:
                return self._cfg.pileup.mapq0_pixel
            else:
                return np.minimum((np.array(qual) / self._cfg.pileup.max_mapq) * 127 + 128, MAX_PIXEL_VALUE) 
        else:
            return self._qual_pixel(qual, self._cfg.pileup.max_mapq)


    def _flatten_image(self, image_tensor, render_channels=False, margin=5):
        if tf.is_tensor(image_tensor):
            image_tensor = image_tensor.numpy()
        
        # TODO: Better combine all the channels into a single image, perhaps ALIGNED, REF_PAIRED_CHANNEL, ALLELE (with mapq as alpha)...
        channels = [ALIGNED_CHANNEL, REF_PAIRED_CHANNEL, ALLELE_CHANNEL]
        combined_image = Image.fromarray(image_tensor[:, :, channels], mode="RGB")  

        if render_channels:
            height, width, num_channels = self._image_shape
            image = Image.new(combined_image.mode,  (width + (num_channels - 1)*(width + margin), 2*height + margin))
            image.paste(combined_image, ((image.width - width) // 2, 0))

            for i in range(num_channels):
                channel_image = Image.fromarray(image_tensor[:, :, [i]*len(channels)], mode=combined_image.mode)
                coord = (i*(width + margin), height + margin)
                image.paste(channel_image, coord)  
            
            return image
        else:
            return combined_image

def image_region(cfg, variant_region: Range) -> Range:
    # TODO: Handle odd-sized variants
    padding = max((cfg.pileup.image_width - variant_region.length + 1) // 2, cfg.pileup.variant_padding)
    return variant_region.expand(padding)

class SingleImageGenerator(ImageGenerator):
    def __init__(self, cfg, num_channels):
        super().__init__(cfg, num_channels=num_channels)
        

    def image_regions(self, variant) -> Range:
        return image_region(self._cfg, variant.reference_region)


    def _add_variant_strip(self, variant: Variant, sample: Sample, pileup: Pileup, region: Range, image_tensor):
        assert variant.num_alt == 1, "Variant strip doesn't support multi-allelic variants"
        image_tensor[:self._cfg.pileup.variant_band_height, :, ALIGNED_CHANNEL] = self._align_pixel(BaseAlignment.MATCH)
        image_tensor[:self._cfg.pileup.variant_band_height, :, MAPQ_CHANNEL] = self._mapq_pixel(self._cfg.pileup.variant_mapq)
        
        ref_zscore, alt_zscore = abs(variant.length_change() / sample.std_insert_size), 0
        image_tensor[:self._cfg.pileup.variant_band_height, :, REF_PAIRED_CHANNEL] = self._zscore_pixel(ref_zscore)
        image_tensor[:self._cfg.pileup.variant_band_height, :, ALT_PAIRED_CHANNEL] = self._zscore_pixel(alt_zscore)
                        
        image_tensor[:self._cfg.pileup.variant_band_height, :, ALLELE_CHANNEL] = 250
        image_tensor[:self._cfg.pileup.variant_band_height, :, STRAND_CHANNEL] = self._strand_to_pixel[Strand.POSITIVE]
        image_tensor[:self._cfg.pileup.variant_band_height, :, BASEQ_CHANNEL] = self._qual_pixel(self._cfg.pileup.variant_baseq, self._cfg.pileup.max_baseq)
      
        # Clip out the region containing the variant
        for col_slice in pileup.region_columns(region, variant.reference_region):
            image_tensor[:self._cfg.pileup.variant_band_height, col_slice, :] = 0

    def render(self, image_tensor, **kwargs) -> Image:
        shape = image_tensor.shape
        assert len(shape) == 3
        return self._flatten_image(image_tensor, **kwargs)


    def generate(self, variant, read_path, sample: Sample, regions=None, realigner=None, alleles: typing.AbstractSet[int]={1}, **kwargs):
        if regions is None:
            regions = self.image_regions(variant)
        elif isinstance(regions, str):
            regions = Range.parse_literal(regions)

        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        if realigner is None:
            realigner = _realigner(variant, sample, reference=self._cfg.reference, flank=self._cfg.pileup.realigner_flank, alleles=alleles)

        image_tensor = self._generate(variant, read_path, sample, regions=regions, realigner=realigner, alleles=alleles, **kwargs)

        # Create consistent image size
        if image_tensor.shape != self.image_shape:
            # resize converts to float directly (however convert_image_dtype assumes floats are in [0-1]) so
            # we use cast instead
            image_tensor = tf.cast(tf.image.resize(image_tensor, self.image_shape[:2]), dtype=tf.uint8).numpy()

        return image_tensor


class SingleHybridImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg, num_channels=6)
        

    def _generate(self, variant, read_path, sample: Sample, regions, realigner, **kwargs):
        assert variant.num_alt == 1
        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)

        # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
        pileup = Pileup(regions)
        
        realigned_reads = []
        straddling_fragments = []

        left_region = variant.left_flank_region(self._cfg.pileup.fetch_flank)  # TODO: Incorporate CIPOS and CIEND?
        right_region = variant.right_flank_region(self._cfg.pileup.fetch_flank)

        for fragment in fragments:
            # Only record the zscore for reads that straddle the event
            if fragment.fragment_straddles(left_region, right_region, min_aligned=self._cfg.pileup.anchor_min_aligned):
                ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
                alt_zscore = _fragment_zscore(sample, fragment.fragment_length, fragment_delta=variant.length_change())
                straddling_fragments.append((fragment, ref_zscore, alt_zscore))
            else:
                ref_zscore = None
                alt_zscore = None

            # At present we render reads based on the original alignment so we only realign fragments that could overlap
            # the image window
            if fragment.reads_overlap(regions):
                realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)
                realigned_reads.append((fragment.read1, realignment))
                realigned_reads.append((fragment.read2, realignment))
                pileup.add_fragment(fragment, allele=realignment.allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        # Add variant strip at the top of the image, clipping out the variant region
        if self._cfg.pileup.variant_band_height > 0:
            self._add_variant_strip(variant, sample, pileup, regions, image_tensor)

        # Add pileup bases to the image
        # This is the slowest portion of image generation as we iterate through every valid pixel in the image, 
        # including sorting each column. Perhaps move to C++?
        read_pixels = image_height - self._cfg.pileup.variant_band_height
        for j, column in enumerate(pileup):
            for i, base in enumerate(column.ordered_bases(order="aligned", max_bases=read_pixels), start=self._cfg.pileup.variant_band_height):
                image_tensor[i, j, ALIGNED_CHANNEL] = self._align_pixel(base.aligned)
                image_tensor[i, j, MAPQ_CHANNEL] = min(base.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE
                image_tensor[i, j, STRAND_CHANNEL] = self._strand_to_pixel[base.strand]

        # Render realigned reads as "full" reads, not as pileup "bases"
        def _filter_reads(realigned_read):
            read, realignment = realigned_read
            # Keep reads in the image window that overlap one of the variant breakpoints
            return read and realignment.allele != AlleleAssignment.AMB and realignment.breakpoint and regions.get_overlap(read) > 0 and realignment.normalized_score > self._cfg.pileup.min_normalized_allele_score
        
        realigned_reads = [read for read in realigned_reads if _filter_reads(read)]
        if len(realigned_reads) > read_pixels:
            realigned_reads = random.sample(realigned_reads, k=read_pixels)
        realigned_reads.sort(key=lambda x: read_start(x[0]))

        for i, (read, realignment) in enumerate(realigned_reads, start=self._cfg.pileup.variant_band_height):
            allele_pixel = self._allele_pixel(realignment)
            _, col_slices = pileup.read_columns(read)
            for col_slice, *_ in col_slices:
                image_tensor[i, col_slice, ALLELE_CHANNEL] = allele_pixel

        # Render fragment insert size as "bars" since straddling reads may not be in the image
        if len(straddling_fragments) > read_pixels:
            straddling_fragments = random.sample(straddling_fragments, k=read_pixels)
        straddling_fragments.sort(key = lambda x: x[0].fragment_start)
        for i, (fragment, ref_zscore, alt_zscore) in enumerate(straddling_fragments, start=self._cfg.pileup.variant_band_height):
            for col_slice in pileup.region_columns(fragment.fragment_region):
                image_tensor[i, col_slice, REF_PAIRED_CHANNEL] = self._zscore_pixel(ref_zscore)
                image_tensor[i, col_slice, ALT_PAIRED_CHANNEL] = self._zscore_pixel(alt_zscore)

        return image_tensor


@dataclass
class _StraddleRegions:
    left_region: Range
    right_region: Range
    event_region: Range = field(init=False)

    def __post_init__(self):
        assert self.left_region.contig == self.right_region.contig
        self.event_region = Range(self.left_region.contig, self.left_region.end, self.right_region.start)


class SingleDepthImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg, num_channels=7)
        

    def _generate(self, variant, read_path, sample: Sample, regions, realigner, ref_seq: str = None, alleles: typing.AbstractSet[int]={1}, **kwargs):
        assert variant.is_deletion, "Currently only deletions supported"
        assert ref_seq is None or len(ref_seq) == regions.length

        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)

        # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
        pileup = ReadPileup(regions)

        straddle_regions = []
        for allele in variant.alt_allele_indices:
            straddle_regions.append(_StraddleRegions(
                variant.left_flank_region(self._cfg.pileup.fetch_flank, allele=allele),
                variant.right_flank_region(self._cfg.pileup.fetch_flank, allele=allele),
            ))
        
        for fragment in fragments:
            # At present we render reads based on the original alignment so we only realign (and track) fragments that could overlap
            # the image window. If we render "insert" bases, then we look if any part of the fragment overlaps the region
            if fragment.fragment_overlaps(regions, read_overlap_only=not self._cfg.pileup.insert_bases):
                realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)
                
                # Prefer possible alternate alleles (i.e. the best "alt") vs. reference straddlers
                ref_zscore, alt_zscore = None, None  # Best is defined as closest to zero
                for length_change, straddle in zip(variant.length_change(allele=None), straddle_regions):
                    allele_alt_zscore = _fragment_zscore(sample, fragment.fragment_length, fragment_delta=length_change)
                    if fragment.fragment_straddles(straddle.left_region, straddle.right_region) and (alt_zscore is None or abs(allele_alt_zscore) < abs(alt_zscore)):
                        ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
                        alt_zscore = allele_alt_zscore
                    elif alt_zscore is None and (fragment.fragment_straddles(straddle.left_region, straddle.event_region) ^ fragment.fragment_straddles(straddle.event_region, straddle.right_region)):
                        ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
                    

                # Render "insert" bases for overlapping fragments without reads in the region (and thus would not
                # otherwise be represented)
                add_insert = self._cfg.pileup.insert_bases and not fragment.reads_overlap(regions)
                pileup.add_fragment(fragment, add_insert=add_insert, ref_seq=ref_seq, allele=realignment, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        # Add variant strip at the top of the image, clipping out the variant region
        if self._cfg.pileup.variant_band_height > 0:
            self._add_variant_strip(variant, sample, pileup, regions, image_tensor)

        # Add pileup bases to the image (downsample reads based on simple coverage-based heuristic)
        row_idxs = np.full((regions.length,), self._cfg.pileup.variant_band_height)
        max_reads = (regions.length * (image_height - self._cfg.pileup.variant_band_height)) // sample.read_length
        for read in pileup.overlapping_reads(regions, max_reads=max_reads):
            for col_slice, aligned, read_slice in pileup.read_columns(regions, read, ref_seq):
                col_idxs = range(col_slice.start, col_slice.stop)
                image_tensor[row_idxs[col_slice], col_idxs, ALIGNED_CHANNEL] = self._align_pixel(aligned)
                image_tensor[row_idxs[col_slice], col_idxs, MAPQ_CHANNEL] = self._mapq_pixel(read.mapq)
                image_tensor[row_idxs[col_slice], col_idxs, REF_PAIRED_CHANNEL] = self._zscore_pixel(read.ref_zscore)
                image_tensor[row_idxs[col_slice], col_idxs, ALT_PAIRED_CHANNEL] = self._zscore_pixel(read.alt_zscore)
                image_tensor[row_idxs[col_slice], col_idxs, ALLELE_CHANNEL] = self._allele_pixel(read.allele)
                image_tensor[row_idxs[col_slice], col_idxs, STRAND_CHANNEL] = self._strand_to_pixel[read.strand]
                image_tensor[row_idxs[col_slice], col_idxs, BASEQ_CHANNEL] = self._qual_pixel(read.baseq(read_slice), self._cfg.pileup.max_baseq)
                
                # Increment the 'current' row for the bases we just added to the pileup, overwrite the last row if we exceed
                # the maximum coverage
                row_idxs[col_slice] = np.clip(row_idxs[col_slice] + 1, self._cfg.pileup.variant_band_height, image_height - 1) 


        return image_tensor


class SingleFragmentImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg, num_channels=6)
        assert self._cfg.pileup.variant_band_height == 0, "Generator doesn't support a variant band"
    
    def _generate(self, variant, read_path, sample: Sample, regions, realigner, **kwargs):        
        assert variant.num_alt == 1
        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)
    
        pileup = FragmentPileup([regions])
        for fragment in fragments:
            # At present we render reads based on the original alignment so skip any fragment with no reads in the region
            # TODO: Eliminate the proper pairing requirement
            if not fragment.is_properly_paired or not fragment.fragment_overlaps(regions, read_overlap_only=False):
                continue
            
            # Realign reads
            realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)

            # Determine the Z-score for all reads 
            ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length, fragment_delta=variant.length_change())
        
            pileup.add_fragment(fragment, allele=realignment.allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        read_pixels = image_height
        
        # Get and potentially downsample fragments overlapping the image
        window_fragments = pileup.overlapping_fragments(regions, max_length=read_pixels)
        for i, fragment in enumerate(window_fragments):
            # Render insert size spanning the entire fragment        
            for col_slice in pileup.fragment_columns(regions, fragment):
                image_tensor[i, col_slice, REF_PAIRED_CHANNEL] = self._zscore_pixel(fragment.ref_zscore)
                image_tensor[i, col_slice, ALT_PAIRED_CHANNEL] = self._zscore_pixel(fragment.alt_zscore)
            
            for read in fragment.reads:
                # Render pixels for component reads
                for col_slice, aligned, _ in pileup.read_columns(regions, read):
                    image_tensor[i, col_slice, ALIGNED_CHANNEL] = self._align_pixel(aligned)
                    image_tensor[i, col_slice, ALLELE_CHANNEL] = self._allele_to_pixel[fragment.allele]
                    image_tensor[i, col_slice, MAPQ_CHANNEL] = min(read.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE
                    image_tensor[i, col_slice, STRAND_CHANNEL] = self._strand_pixel(read)

        return image_tensor


class WindowedImageGenerator(ImageGenerator):
    def __init__(self, cfg, num_channels):
        super().__init__(cfg, num_channels=num_channels)

    @property
    def image_shape(self):
        windows = 2 if self._cfg.pileup.get("breakpoints_only", False) else None
        return (windows,) + self._image_shape

    def image_regions(self, variant) -> typing.List[Range]:
        if self._cfg.pileup.get("breakpoints_only", False):
            return variant.window_regions(self._cfg.pileup.image_width, flank_windows=0, window_interior=False)
        else:
            return variant.window_regions(self._cfg.pileup.image_width, self._cfg.pileup.flank_windows)

    def render(self, image_tensor) -> Image:
        shape = image_tensor.shape
        assert len(shape) == 4
        windows, height, width, _ = shape
        composite_image = Image.new("RGB", (windows*width, height))
        for i in range(windows):
            sub_image = self._flatten_image(image_tensor[i])
            composite_image.paste(sub_image, (i*width, 0))
        return composite_image

    def generate(self, variant, read_path, sample: Sample, regions=None, realigner=None, **kwargs):
        assert variant.num_alt == 1
        if regions is None:
            regions = self.image_regions(variant)
        for region in regions[1:]:
            assert region.length == regions[0].length, "All regions must have identical lengths"

        if realigner is None:
            realigner = _realigner(variant, sample, reference=self._cfg.reference, flank=self._cfg.pileup.realigner_flank)

        return self._generate(variant, read_path, sample, regions=regions, realigner=realigner, **kwargs)


class WindowedReadImageGenerator(WindowedImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg, num_channels=6)


    def _generate(self, variant, read_path, sample: Sample, regions, realigner, **kwargs):         
        assert self.image_shape[-1] == 6
        tensor_shape = (len(regions),) + self.image_shape[1:]
        image_tensor = np.zeros(tensor_shape, dtype=np.uint8)

        # Determine the outer extent of all windows
        region = functools.reduce(lambda a, b: a.union(b), regions)

        fragments = _fetch_reads(read_path, region.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)

        pileup = ReadPileup(regions)

        for fragment in fragments:
            # At present we render reads based on the original alignment so skip any read that doesn't map to the image
            # region
            if not fragment.reads_overlap(region):
                continue
            
            # Realign reads
            realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)

            # Determine the Z-score for all reads 
            ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length, fragment_delta=variant.length_change())
        
            pileup.add_fragment(fragment, allele=realignment.allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        read_pixels = tensor_shape[1]
        for w, window_region in enumerate(regions):
            # Get and potential downsample reads overlapping this window
            window_reads = pileup.overlapping_reads(window_region, max_reads=read_pixels)
            for i, read in enumerate(window_reads):
                for col_slice, aligned, _ in pileup.read_columns(window_region, read):
                    image_tensor[w, i, col_slice, ALIGNED_CHANNEL] = self._align_pixel(aligned)
                    image_tensor[w, i, col_slice, REF_PAIRED_CHANNEL] = self._zscore_pixel(read.ref_zscore)
                    image_tensor[w, i, col_slice, ALT_PAIRED_CHANNEL] = self._zscore_pixel(read.alt_zscore)
                    image_tensor[w, i, col_slice, ALLELE_CHANNEL] = self._allele_to_pixel[read.allele]
                    image_tensor[w, i, col_slice, MAPQ_CHANNEL] = self._qual_pixel(read.mapq, self._cfg.pileup.max_mapq)
                    image_tensor[w, i, col_slice, STRAND_CHANNEL] = self._strand_to_pixel[read.strand]

        return image_tensor
   



def _genotype_to_label(genotype, alleles: typing.AbstractSet[int]={1}):
    # TODO: Handle no-calls
    count = 0
    for gt in genotype:
        if gt == -1:
            return None
        elif gt in alleles:
            count += 1
    return count


# Adapted from DeepVariant
def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a list of string / byte."""
    if isinstance(list_of_strings, type(tf.constant(0))):
        list_of_strings = [list_of_strings.numpy()]  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))


def _int_feature(list_of_ints):
    """Returns a int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):
    """Returns a float_list from a list of int / bool."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def make_variant_example(cfg, variant: Variant, read_path: str, sample: Sample, label=None, simulate=False, generator=None, random_variants=None, alleles: typing.AbstractSet[int]={1}, addl_features={}):
    assert 1 <= len(alleles) <= min(variant.num_alt, 2)

    if generator is None:
        generator = hydra.utils.instantiate(cfg.generator, cfg=cfg)
    if cfg.simulation.sample_ref and random_variants is None:
        random_variants = RandomVariants(cfg.reference, cfg.simulation.sample_exclude_bed)

    example_regions = generator.image_regions(variant)
    
    # Construct realigner once for all images for this variant, and if rendering match/mismatch bases, 
    # obtain relevant reference sequence for this variant, including IUPAC bases if SNV data is provided
    if cfg.pileup.get("render_snv", False):
        realigner = _realigner(variant, sample, reference=cfg.reference, flank=cfg.pileup.realigner_flank, snv_vcf_path=cfg.pileup.snv_vcf_input, alleles=alleles)
        ref_seq = _reference_sequence(cfg.reference, example_regions, snv_vcf_path=cfg.pileup.snv_vcf_input)
    else:
        realigner = _realigner(variant, sample, reference=cfg.reference, flank=cfg.pileup.realigner_flank, alleles=alleles)
        ref_seq = None
   
    # Construct image for "real" data
    image_tensor = generator.generate(
        variant, read_path, sample, realigner=realigner, regions=example_regions, ref_seq=ref_seq,
    )

    feature = {
        "variant/encoded": _bytes_feature([variant.as_proto().SerializeToString()]),
        "image/shape": _int_feature(image_tensor.shape),
        "image/encoded": _bytes_feature(tf.io.serialize_tensor(image_tensor)),
    }

    if label is not None:
        feature["label"] = _int_feature([label])

    replicates = cfg.simulation.replicates
    if simulate and cfg.simulation.replicates > 0:
        # A 5/6-D tensor for simulated images (AC, REPLICATES) + (WINDOWS?, ROW, COLS, CHANNELS)
        feature["sim/images/shape"] = _int_feature((3, replicates) + image_tensor.shape)

        # Generate synthetic training images
        ac_encoded_images = [None] * 3
    
        # If we are augmenting the simulated data, use the provided statistics for the first example, so it
        # will hopefully be similar to the real data and then augment the remaining replicates
        if cfg.simulation.augment:
            repl_samples = augment_samples(sample, replicates, keep_original=True)
        else:
            repl_samples = [sample] * cfg.simulation.replicates

        for allele_count in range(3):
            with tempfile.TemporaryDirectory() as tempdir:
                # Generate the FASTA file for this zygosity
                if allele_count == 0:
                    fasta_alleles = (0, 0)
                elif allele_count == 1:
                    # Use first allele in multi-allelic contexts (where heterozygous is ill-defined)
                    fasta_alleles = (0, next(iter(alleles)))
                elif allele_count == 2:
                    fasta_alleles = sorted(alleles) * (allele_count // len(alleles))
                haplotypes = len(set(fasta_alleles)) # Number of distinct haplotypes

                fasta_path, ref_contig, alt_contig = variant.synth_fasta(
                    reference_fasta=cfg.reference, alleles=fasta_alleles, flank=cfg.pileup.realigner_flank, dir=tempdir
                )

                if cfg.simulation.gnomad_norm_covg:
                    gnomad_covg_path = variant.gnomad_coverage_profile(cfg.simulation.gnomad_covg, ref_contig=ref_contig, alt_contig=alt_contig, flank=cfg.pileup.realigner_flank, dir=tempdir)
                else:
                    gnomad_covg_path = None

                # Generate and image synthetic bam files
                repl_encoded_images = []
                
                sim_replicates = replicates if allele_count != 0 or not cfg.simulation.sample_ref else 0         
                for i, repl_sample in enumerate(repl_samples[:sim_replicates]):
                    try:
                        sample_coverage = repl_sample.chrom_mean_coverage(variant.contig) if cfg.simulation.chrom_norm_covg else repl_sample.mean_coverage
                        replicate_bam_path = simulate_variant_sequencing(
                            fasta_path,
                            sample_coverage / haplotypes,
                            repl_sample,
                            reference=cfg.reference,
                            shared_reference=cfg.shared_reference,
                            dir=tempdir,
                            stats_path=cfg.stats_path if cfg.simulation.gc_norm_covg else None,
                            gnomad_covg_path=gnomad_covg_path,
                        )
                    except ValueError:
                        logging.error("Failed to synthesize data for %s with AC=%d", str(variant), allele_count)
                        raise

                    synth_image_tensor = generator.generate(variant, replicate_bam_path, repl_sample, realigner=realigner, regions=example_regions, ref_seq=ref_seq)
                    repl_encoded_images.append(synth_image_tensor)

                    if not OmegaConf.is_missing(cfg.simulation, "save_sim_bam_dir"):
                        sim_bam_path=os.path.join(cfg.simulation.save_sim_bam_dir, f"{variant.name}_{allele_count}_{i}.bam")
                        shutil.copy(replicate_bam_path, sim_bam_path)
                        shutil.copy(f"{replicate_bam_path}.bai", f"{sim_bam_path}.bai")

                # Fill remaining images with sampled reference variants
                if allele_count == 0 and cfg.simulation.sample_ref:
                    for random_variant in random_variants.generate(variant, replicates - sim_replicates):          
                        # TODO: Should we also render match/mismatch when sampling?
                        random_variant_regions = generator.image_regions(random_variant)
                        synth_image_tensor = generator.generate(random_variant, read_path, sample, regions=random_variant_regions)
                        repl_encoded_images.append(synth_image_tensor)

                # Stack all of the image replicates into a tensor
                ac_encoded_images[allele_count] = np.stack(repl_encoded_images)

        # Stack the replicated images for the 3 genotypes (0/0, 0/1, 1/1) into tensor
        sim_image_tensor = np.stack(ac_encoded_images)
        feature[f"sim/images/encoded"] = _bytes_feature(tf.io.serialize_tensor(sim_image_tensor))

        # Add any additional (extension) features
        feature.update(addl_features)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_vcf_examples(
    cfg,
    vcf_path: str,
    read_path: str,
    sample: Sample,
    sample_or_label=None,
    region: str = None,
    num_shards: int = 1,
    index: int = 0,
    **kwargs,
):
    # Create image generator based on current configuration
    generator = hydra.utils.instantiate(cfg.generator, cfg)
    
    # Prepare random variant generator once (if specified)
    if cfg.simulation.sample_ref:
        random_variants = RandomVariants(cfg.reference, cfg.simulation.sample_exclude_bed)
    else:
        random_variants = None

    with pysam.VariantFile(vcf_path) as vcf_file:
        # Prepare function to extract genotype label
        if sample_or_label is not None and isinstance(sample_or_label, str):
            samples = vcf_file.header.samples
            sample_index = next((i for i, s in enumerate(samples) if s == sample_or_label), -1,)
            if sample_index == -1:
                raise ValueError("Sample identifier is not present in the file")
            logging.info("Using %s genotypes as labels (VCF sample index %d)", sample_or_label, sample_index)
            vcf_file.subset_samples([sample_or_label])
            # After subsetting there is only one sample (so index is always 0)
            label_extractor = lambda variant: _genotype_to_label(variant.genotype_indices(0))
        else:
            if sample_or_label is not None:
                logging.info("Using fixed AC=%d as label", sample_or_label)
            vcf_file.subset_samples([])  # Drop all samples
            label_extractor = lambda variant: sample_or_label        

        if region:
            query_range = Range.parse_literal(region)
            variant_iter = vcf_file.fetch(**query_range.pysam_fetch)
        else:
            variant_iter = vcf_file

        for i, record in enumerate(variant_iter):
            if i % num_shards == index:
                variant = Variant.from_pysam(record)
                assert variant.is_biallelic(), "Multi-allelic sites not yet supported"

                # To avoid duplicated entries, only generate images for variants that start within region
                if region and not query_range.contains(variant.start):
                    continue

                label = label_extractor(variant)
                yield make_variant_example(cfg, variant, read_path, sample, label=label, generator=generator, random_variants=random_variants, **kwargs)
           


def _filename_to_compression(filename: str) -> str:
    if filename.endswith(".gz"):
        return "GZIP"
    else:
        return None


def vcf_to_tfrecords(
    cfg,
    vcf_path: str,
    read_path: str,
    output_path: str,
    sample: Sample,
    sample_or_label=None,
    simulate=False,
    progress_bar=False,
):
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        logging.info("Initializing ray with %d threads", cfg.threads)
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False, object_store_memory=8*1024*1024*1024, _redis_max_memory=1024*1024*1024)

        def _vcf_shard(num_shards: int, index: int) -> typing.Iterator[tf.train.Example]:
            try:
                # Try to reduce the number of threads TF creates since we are running multiple instances of TF via Ray
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except RuntimeError:
                pass
            
            yield from make_vcf_examples(cfg, vcf_path, read_path, sample, sample_or_label=sample_or_label, simulate=simulate, num_shards=num_shards, index=index)

        # Create parallel iterators. We use a partial wrapper because the generator alone can't be be pickled.
        # There is a warning due to an internal exception bubbling up that seems to have to do with changes
        # to Python generator handling. Not clear how to fix that warning...
        it = ray.util.iter.from_iterators([functools.partial(_vcf_shard, cfg.threads, i) for i in range(cfg.threads)])
        
        with tf.io.TFRecordWriter(output_path, _filename_to_compression(output_path)) as file_writer:
            for example in tqdm(it.gather_async(), desc="Generating variant images", disable=not progress_bar):
                file_writer.write(example.SerializeToString()) 


def _features_variant(features):
    return npsv2_pb2.StructuralVariant.FromString(features["variant/encoded"].numpy())


def _example_variant(example):
    encoded_variant = example.features.feature["variant/encoded"].bytes_list.value[0]
    return npsv2_pb2.StructuralVariant.FromString(encoded_variant)


def _example_image(example):
    image_data = tf.io.parse_tensor(example.features.feature["image/encoded"].bytes_list.value[0], tf.uint8).numpy()
    return image_data


def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value)


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _example_sim_images_shape(example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value)
    else:
        return (3, 0, None, None, None)


def _example_sim_images(example):
    image_data = tf.io.parse_tensor(
        example.features.feature["sim/images/encoded"].bytes_list.value[0], tf.uint8
    ).numpy()
    return image_data


def _extract_metadata_from_first_example(filename):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())

    image_shape = _example_image_shape(example)
    ac, replicates, *sim_image_shape = _example_sim_images_shape(example)
    if replicates > 0:
        assert ac == 3, "Incorrect number of genotypes in simulated data"
        assert image_shape == tuple(sim_image_shape), "Simulated and actual image shapes don't match"

    return image_shape, replicates


def features_to_image(cfg, features, out_path: str, with_simulations=False, margin=10, max_replicates=1, render_channels=False):
    generator = hydra.utils.instantiate(cfg.generator, cfg)

    image_tensor = features["image"]
    real_image = generator.render(image_tensor, render_channels=render_channels)

    _, replicates, *_ = features["sim/images"].shape if with_simulations and "sim/images" in features else (3, 0)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode,  (width + 2 * (width + margin), height + replicates * (height + margin)))
        image.paste(real_image, (width + margin, 0))

        synth_tensor = features["sim/images"]
        for ac in range(3):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[ac, repl]
                synth_image = generator.render(synth_image_tensor, render_channels=render_channels)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def example_to_image(cfg, example: tf.train.Example, out_path: str, with_simulations=False, **kwargs):
    features = {
        "image": _example_image(example),
    }
    _, replicates, *_ = _example_sim_images_shape(example)
    if with_simulations and replicates > 0:
        features["sim/images"] = _example_sim_images(example)
    
    features_to_image(cfg, features, out_path, with_simulations=with_simulations and replicates > 0, **kwargs)


def load_example_dataset(filenames, with_label=False, with_simulations=False, num_parallel_reads=None) -> tf.data.Dataset:
    if isinstance(filenames, str):
        filenames = [filenames]
    assert len(filenames) > 0

    # Extract image shape from the first example
    shape, replicates = _extract_metadata_from_first_example(filenames[0])

    proto_features = {
        "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
    }
    if with_label:
        proto_features["label"] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    if with_simulations and replicates > 0:
        proto_features.update(
            {
                "sim/images/shape": tf.io.FixedLenFeature(shape=(len(shape) + 2,), dtype=tf.int64),
                "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            }
        )

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            "variant/encoded": parsed_features["variant/encoded"],
            "image": tf.io.parse_tensor(parsed_features["image/encoded"], tf.uint8),
        }
        if with_simulations:
            features["sim/images"] = tf.io.parse_tensor(parsed_features["sim/images/encoded"], tf.uint8)

        if with_label:
            return features, parsed_features["label"]
        else:
            return features, None

    compression = _filename_to_compression(filenames[0])
    num_parallel_calls = num_parallel_reads if num_parallel_reads is None or num_parallel_reads == tf.data.experimental.AUTOTUNE else min(len(filenames), num_parallel_reads)
    return tf.data.Dataset.from_tensor_slices(filenames).interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type=compression).map(_process_input, num_parallel_calls=1), cycle_length=len(filenames), num_parallel_calls=num_parallel_calls)

