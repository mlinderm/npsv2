import datetime, functools, logging, os, random, subprocess, sys, tempfile, typing
import numpy as np
import pysam
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import ray
import hydra
from omegaconf import OmegaConf

from .variant import Variant
from .range import Range
from .pileup import Pileup, FragmentTracker, AlleleAssignment, BaseAlignment, ReadPileup, FragmentPileup, read_start
from . import npsv2_pb2
from .realigner import FragmentRealigner, realign_fragment, AlleleRealignment
from .simulation import RandomVariants, simulate_variant_sequencing, augment_samples
from .sample import Sample


MAX_PIXEL_VALUE = 254.0  # Adapted from DeepVariant


def _fragment_zscore(sample: Sample, fragment_length: int):
    return (fragment_length - sample.mean_insert_size) / sample.std_insert_size


def _realigner(variant, sample: Sample, reference, flank=1000):
    with tempfile.TemporaryDirectory() as dir:
        fasta_path, ref_contig, alt_contig = variant.synth_fasta(reference_fasta=reference, dir=dir, flank=flank)
        
        # Convert breakpoints to list of strings to be passed into the C++ side
        breakpoints = variant.ref_breakpoints(flank, contig=ref_contig) + variant.alt_breakpoints(flank, contig=alt_contig)
        breakpoints = [tuple(map(lambda x: str(x) if x else "", breakpoints))]

        return FragmentRealigner(fasta_path, breakpoints, sample.mean_insert_size, sample.std_insert_size)


def _fetch_reads(read_path: str, fetch_region: Range) -> FragmentTracker:
    fragments = FragmentTracker()
    with pysam.AlignmentFile(read_path) as alignment_file:
        for read in alignment_file.fetch(
            contig=fetch_region.contig, start=fetch_region.start, stop=fetch_region.end
        ):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            fragments.add_read(read)
    return fragments

class ImageGenerator:
    def __init__(self, cfg):
        self._cfg = cfg
        self._image_shape = (self._cfg.pileup.image_height, self._cfg.pileup.image_width, self._cfg.pileup.num_channels)

        # Helper dictionaries
        self._aligned_to_pixel = {
            BaseAlignment.ALIGNED: self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.SOFT_CLIP: self._cfg.pileup.soft_clip_base_pixel,
        }

        self._allele_to_pixel = {
            AlleleAssignment.AMB: self._cfg.pileup.amb_allele_pixel,
            AlleleAssignment.REF: self._cfg.pileup.ref_allele_pixel,
            AlleleAssignment.ALT: self._cfg.pileup.alt_allele_pixel,
        }

    @property
    def image_shape(self):
        return self._image_shape


    def _zscore_pixel(self, zscore):
        return np.clip(
            self._cfg.pileup.insert_size_mean_pixel + zscore * self._cfg.pileup.insert_size_sd_pixel, 1, MAX_PIXEL_VALUE
        )

    def _allele_pixel(self, realignment: AlleleRealignment):
        # if self._cfg.pileup.weight_allele:
        #     # STOPPPED HERE
        #     assert self._cfg.pileup.ref_allele_pixel < self._cfg.pileup.amb_allele_pixel < self._cfg.pileup.alt_allele_pixel
        #     if realignment.allele == AlleleAssignment.REF:
        #         return min(self._cfg.pileup.ref_allele_pixel + 2*abs(realignment.normalized_score), self._cfg.pileup.amb_allele_pixel)
        #     elif realignment.allele == AlleleAssignment.ALT:
        #         print(2*abs(realignment.normalized_score))
        #         print(max(self._cfg.pileup.alt_allele_pixel - 2*abs(realignment.normalized_score), self._cfg.pileup.amb_allele_pixel))
        #         abs(realignment.normalized_score) / self._cfg
        #         return max(self._cfg.pileup.alt_allele_pixel - abs(realignment.normalized_score) / self._cfg., self._cfg.pileup.amb_allele_pixel)
        #     else:
        #         return self._cfg.pileup.amb_allele_pixel
        # else:
        #     return self._allele_to_pixel[realignment.allele]
        return self._allele_to_pixel[realignment.allele]

    def _flatten_image(self, image_tensor):
        if tf.is_tensor(image_tensor):
            image_tensor = image_tensor.numpy()
        # TODO: Combine all the channels into a single image, perhaps BASE, INSERT_SIZE, ALLELE (with
        # mapq as alpha)...
        channels = [self._cfg.pileup.aligned_channel, self._cfg.pileup.ref_paired_channel, self._cfg.pileup.allele_channel]
        #channels = 3*[self._cfg.pileup.allele_channel]
        return Image.fromarray(image_tensor[:, :, channels], mode="RGB")    


class SingleImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        

    def image_regions(self, variant) -> Range:
        # TODO: Handle odd-sized variants
        # Construct a single "padded" region to render as the pileup image
        variant_region = variant.reference_region
        padding = max((self._cfg.pileup.image_width - variant_region.length + 1) // 2, self._cfg.pileup.variant_padding)
        return variant_region.expand(padding)


    def _add_variant_strip(self, variant: Variant, sample: Sample, pileup: Pileup, image_tensor):
        image_tensor[:self._cfg.pileup.variant_band_height, :, self._cfg.pileup.aligned_channel] = self._cfg.pileup.aligned_base_pixel
        image_tensor[:self._cfg.pileup.variant_band_height, :, self._cfg.pileup.mapq_channel] = min(self._cfg.pileup.variant_mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE
        
        ref_zscore, alt_zscore = abs(variant.length_change() / sample.std_insert_size), 0
        image_tensor[:self._cfg.pileup.variant_band_height, :, self._cfg.pileup.ref_paired_channel] = self._zscore_pixel(ref_zscore)
        image_tensor[:self._cfg.pileup.variant_band_height, :, self._cfg.pileup.alt_paired_channel] = self._zscore_pixel(alt_zscore)

        # Clip out the region containing the variant
        for col_slice in pileup.region_columns(variant.reference_region):
            image_tensor[:self._cfg.pileup.variant_band_height, col_slice, :] = 0

    def render(self, image_tensor) -> Image:
        shape = image_tensor.shape
        assert len(shape) == 3
        return self._flatten_image(image_tensor)


    def generate(self, variant, read_path, sample: Sample, regions=None, realigner=None):
        if regions is None:
            regions = self.image_regions(variant)
        elif isinstance(regions, str):
            regions = Range.parse_literal(regions)

        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        if realigner is None:
            realigner = _realigner(variant, sample, reference=self._cfg.reference, flank=self._cfg.pileup.realigner_flank)

        image_tensor = self._generate(variant, read_path, sample, regions=regions, realigner=realigner)

        # Create consistent image size
        if image_tensor.shape != self.image_shape:
            # resize converts to float directly (however convert_image_dtype assumes floats are in [0-1]) so
            # we use cast instead
            image_tensor = tf.cast(tf.image.resize(image_tensor, self.image_shape[:2]), dtype=tf.uint8).numpy()

        return image_tensor


class SingleHybridImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        

    def _generate(self, variant, read_path, sample: Sample, regions, realigner):
        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank))

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
                alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
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
            self._add_variant_strip(variant, sample, pileup, image_tensor)

        # Add pileup bases to the image
        # This is the slowest portion of image generation as we iterate through every valid pixel in the image, 
        # including sorting each column. Perhaps move to C++?
        read_pixels = image_height - self._cfg.pileup.variant_band_height
        for j, column in enumerate(pileup):
            for i, base in enumerate(column.ordered_bases(order="aligned", max_bases=read_pixels), start=self._cfg.pileup.variant_band_height):
                image_tensor[i, j, self._cfg.pileup.aligned_channel] = self._aligned_to_pixel[base.aligned]
                image_tensor[i, j, self._cfg.pileup.mapq_channel] = min(base.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE


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
            for col_slice, _ in col_slices:
                image_tensor[i, col_slice, self._cfg.pileup.allele_channel] = allele_pixel

        # Render fragment insert size as "bars" since straddling reads may not be in the image
        if len(straddling_fragments) > read_pixels:
            straddling_fragments = random.sample(straddling_fragments, k=read_pixels)
        straddling_fragments.sort(key = lambda x: x[0].fragment_start)
        for i, (fragment, ref_zscore, alt_zscore) in enumerate(straddling_fragments, start=self._cfg.pileup.variant_band_height):
            for col_slice in pileup.region_columns(fragment.fragment_region):
                image_tensor[i, col_slice, self._cfg.pileup.ref_paired_channel] = self._zscore_pixel(ref_zscore)
                image_tensor[i, col_slice, self._cfg.pileup.alt_paired_channel] = self._zscore_pixel(alt_zscore)

        return image_tensor


class SingleDepthImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        

    def _generate(self, variant, read_path, sample: Sample, regions, realigner):
        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank))

        # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
        pileup = Pileup(regions)
        
        realigned_reads = []

        left_region = variant.left_flank_region(self._cfg.pileup.fetch_flank)  # TODO: Incorporate CIPOS and CIEND?
        right_region = variant.right_flank_region(self._cfg.pileup.fetch_flank)

        for fragment in fragments:
            # At present we render reads based on the original alignment so we only realign (and trakc) fragments that could overlap
            # the image window
            if fragment.reads_overlap(regions):
                realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)
                realigned_reads.append((fragment.read1, realignment))
                realigned_reads.append((fragment.read2, realignment))
                
                # A potential limitation is that if the anchoring read is not in the window, we might not capture that
                # information in the window
                ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
                alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())

                pileup.add_fragment(fragment, allele=realignment, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        # Add variant strip at the top of the image, clipping out the variant region
        if self._cfg.pileup.variant_band_height > 0:
            self._add_variant_strip(variant, sample, pileup, image_tensor)

        # Add pileup bases to the image
        # This is the slowest portion of image generation as we iterate through every valid pixel in the image, 
        # including sorting each column. Perhaps move to C++?
        read_pixels = image_height - self._cfg.pileup.variant_band_height
        for j, column in enumerate(pileup):
            for i, base in enumerate(column.ordered_bases(order="read_start", max_bases=read_pixels), start=self._cfg.pileup.variant_band_height):
                image_tensor[i, j, self._cfg.pileup.aligned_channel] = self._aligned_to_pixel[base.aligned]
                image_tensor[i, j, self._cfg.pileup.mapq_channel] = min(base.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE
                image_tensor[i, j, self._cfg.pileup.ref_paired_channel] = self._zscore_pixel(base.ref_zscore)
                image_tensor[i, j, self._cfg.pileup.alt_paired_channel] = self._zscore_pixel(base.alt_zscore)
                image_tensor[i, j, self._cfg.pileup.allele_channel] = self._allele_pixel(base.allele)

        return image_tensor


class SingleFragmentImageGenerator(SingleImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert self._cfg.pileup.variant_band_height == 0, "Generator doesn't support a variant band"
    
    def _generate(self, variant, read_path, sample: Sample, regions, realigner):        
        image_height, _, num_channels = self.image_shape
        image_tensor = np.zeros((image_height, regions.length, num_channels), dtype=np.uint8)

        fragments = _fetch_reads(read_path, regions.expand(self._cfg.pileup.fetch_flank))
    
        pileup = FragmentPileup([regions])
        for fragment in fragments:
            # At present we render reads based on the original alignment so skip any fragment with no reads in the region
            # TODO: Eliminate the proper pairing requirement
            if not fragment.is_properly_paired or not fragment.reads_overlap(regions, min_overlap=1):
                continue
            
            # Realign reads
            realignment = realign_fragment(realigner, fragment, assign_delta=self._cfg.pileup.assign_delta)

            # Determine the Z-score for all reads 
            ref_zscore = _fragment_zscore(sample, fragment.fragment_length)
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
        
            pileup.add_fragment(fragment, allele=realignment.allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        read_pixels = image_height
        
        # Get and potentially downsample fragments overlapping the image
        window_fragments = pileup.overlapping_fragments(regions, max_length=read_pixels)
        for i, fragment in enumerate(window_fragments):
            # Render insert size spanning the entire fragment        
            for col_slice in pileup.fragment_columns(regions, fragment):
                image_tensor[i, col_slice, self._cfg.pileup.ref_paired_channel] = self._zscore_pixel(fragment.ref_zscore)
                image_tensor[i, col_slice, self._cfg.pileup.alt_paired_channel] = self._zscore_pixel(fragment.alt_zscore)
            
            for read in fragment.reads:
                # Render pixels for component reads
                for col_slice, aligned in pileup.read_columns(regions, read):
                    image_tensor[i, col_slice, self._cfg.pileup.aligned_channel] = self._aligned_to_pixel[aligned]
                    image_tensor[i, col_slice, self._cfg.pileup.allele_channel] = self._allele_to_pixel[fragment.allele]
                    image_tensor[i, col_slice, self._cfg.pileup.mapq_channel] = min(read.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE

        return image_tensor


class WindowedReadImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)

    
    @property
    def image_shape(self):
        return (None,) + self._image_shape


    def image_regions(self, variant) -> typing.List[Range]:
        return variant.window_regions(self._cfg.pileup.image_width, self._cfg.pileup.flank_windows)

        # TODO: Handle odd-sized variants
        # Construct a single "padded" region to render as the pileup image
        variant_region = variant.reference_region
        padding = max((self._cfg.pileup.image_width - variant_region.length + 1) // 2, self._cfg.pileup.variant_padding)
        return variant_region.expand(padding)


    def render(self, image_tensor) -> Image:
        shape = image_tensor.shape
        assert len(shape) == 4
        windows, height, width, _ = shape
        composite_image = Image.new("RGB", (windows*width, height))
        for i in range(windows):
            sub_image = self._flatten_image(image_tensor[i])
            composite_image.paste(sub_image, (i*width, 0))
        return composite_image


    def generate(self, variant, read_path, sample: Sample, regions=None, realigner=None):        
        cfg = self._cfg # TODO: Enable local override of configuration
        if regions is None:
            regions = self.image_regions(variant)
        for region in regions[1:]:
            assert region.length == regions[0].length, "All regions must have identical lengths"
        
        # Determine the outer extent of all windows
        region = functools.reduce(lambda a, b: a.union(b), regions)

        tensor_shape = (len(regions),) + self.image_shape[1:]
        image_tensor = np.zeros(tensor_shape, dtype=np.uint8)

        if realigner is None:
            realigner = _realigner(variant, sample, reference=self._cfg.reference, flank=self._cfg.pileup.realigner_flank)

        fragments = _fetch_reads(read_path, region.expand(self._cfg.pileup.fetch_flank))

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
            alt_zscore = _fragment_zscore(sample, fragment.fragment_length + variant.length_change())
        
            pileup.add_fragment(fragment, allele=realignment.allele, ref_zscore=ref_zscore, alt_zscore=alt_zscore)

        read_pixels = tensor_shape[1]
        for w, window_region in enumerate(regions):
            # Get and potential downsample reads overlapping this window
            window_reads = pileup.overlapping_reads(window_region, max_reads=read_pixels)
            for i, read in enumerate(window_reads):
                for col_slice, aligned in pileup.read_columns(window_region, read):
                    image_tensor[w, i, col_slice, self._cfg.pileup.aligned_channel] = self._aligned_to_pixel[aligned]
                    image_tensor[w, i, col_slice, self._cfg.pileup.ref_paired_channel] = np.clip(
                        self._cfg.pileup.insert_size_mean_pixel + read.ref_zscore * self._cfg.pileup.insert_size_sd_pixel, 1, MAX_PIXEL_VALUE
                    )
                    image_tensor[w, i, col_slice, self._cfg.pileup.alt_paired_channel] = np.clip(
                        self._cfg.pileup.insert_size_mean_pixel + read.alt_zscore * self._cfg.pileup.insert_size_sd_pixel, 1, MAX_PIXEL_VALUE
                    )
                    image_tensor[w, i, col_slice, self._cfg.pileup.allele_channel] = self._allele_to_pixel[read.allele]
                    image_tensor[w, i, col_slice, self._cfg.pileup.mapq_channel] = min(read.mapq / self._cfg.pileup.max_mapq, 1.0) * MAX_PIXEL_VALUE

        return image_tensor
   



def _genotype_to_label(genotype, alleles={1}):
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


def make_variant_example(cfg, variant: Variant, read_path: str, sample: Sample, label=None, simulate=False, generator=None):
    if generator is None:
        generator = hydra.utils.instantiate(cfg.generator, cfg=cfg)

    # Construct realigner once for all images for this variant
    realigner = _realigner(variant, sample, reference=cfg.reference, flank=cfg.pileup.realigner_flank)

    # Construct image for "real" data
    example_regions = generator.image_regions(variant)
    image_tensor = generator.generate(
        variant, read_path, sample, realigner=realigner, regions=example_regions
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
        # will hopefully be similar to the real data and the augment the remaining replicates
        if cfg.simulation.augment:
            repl_samples = augment_samples(sample, replicates, keep_original=True)
        else:
            repl_samples = [sample] * cfg.simulation.replicates

        for allele_count in range(3):
            with tempfile.TemporaryDirectory() as tempdir:
                # Generate the FASTA file for this zygosity
                fasta_path, ref_contig, alt_contig = variant.synth_fasta(
                    reference_fasta=cfg.reference, ac=allele_count, flank=cfg.pileup.realigner_flank, dir=tempdir
                )

                # Generate and image synthetic bam files
                repl_encoded_images = []
                
                sim_replicates = replicates if allele_count != 0 or not cfg.simulation.sample_ref else 0         
                for i in range(sim_replicates):
                    try:
                        replicate_bam_path = simulate_variant_sequencing(
                            fasta_path, allele_count, repl_samples[i], reference=cfg.reference, shared_reference=cfg.shared_reference, dir=tempdir
                        )
                    except ValueError:
                        logging.error("Failed to synthesize data for %s with AC=%d", str(variant), allele_count)
                        raise

                    synth_image_tensor = generator.generate(variant, replicate_bam_path, repl_samples[i], realigner=realigner, regions=example_regions)
                    repl_encoded_images.append(synth_image_tensor)

                # Fill remaining images with sampled reference variants
                if allele_count == 0 and cfg.simulation.sample_ref:
                    for random_variant in random_variants.generate(variant, replicates - sim_replicates):          
                        random_variant_regions = generator.image_regions(random_variant)
                        synth_image_tensor = generator.generate(random_variant, read_path, sample, realigner=realigner, regions=random_variant_regions)
                        repl_encoded_images.append(synth_image_tensor)

                # Stack all of the image replicates into a tensor
                ac_encoded_images[allele_count] = np.stack(repl_encoded_images)

        # Stack the replicated images for the 3 genotypes (0/0, 0/1, 1/1) into tensor
        sim_image_tensor = np.stack(ac_encoded_images)
        feature[f"sim/images/encoded"] = _bytes_feature(tf.io.serialize_tensor(sim_image_tensor))

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
    generator = hydra.utils.instantiate(cfg.generator, cfg=cfg)
    
    with pysam.VariantFile(vcf_path) as vcf_file:
        # Prepare function to extract genotype label
        if sample_or_label is not None and isinstance(sample_or_label, str):
            samples = vcf_file.header.samples
            sample_index = next((i for i, s in enumerate(samples) if s == sample_or_label), -1,)
            if sample_index == -1:
                raise ValueError("Sample identifier is not present in the file")
            logging.info("Using %s genotypes as labels (VCF sample index %d)", sample_or_label, sample_index)
            vcf_file.subset_samples([sample_or_label])
            label_extractor = lambda variant: _genotype_to_label(variant.genotype_indices(sample_index))
        else:
            if sample_or_label is not None:
                logging.info("Using fixed AC=%d as label", sample_or_label)
            vcf_file.subset_samples([])  # Drop all samples
            label_extractor = lambda variant: sample_or_label

        # Prepare random variant generator (if specified)
        if cfg.simulation.sample_ref:
            random_variants = RandomVariants(params.reference, params.exclude_bed)

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
                yield make_variant_example(cfg, variant, read_path, sample, label=label, generator=generator, **kwargs)
           


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

    # We currently just use ray for the CPU-side work, specifically simulating the SVs
    ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=tempfile.gettempdir(), ignore_reinit_error=True, include_dashboard=False)

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


def features_to_image(cfg, features, out_path: str, with_simulations=False, margin=10, max_replicates=1):
    generator = hydra.utils.instantiate(cfg.generator, cfg=cfg)

    image_tensor = features["image"]
    real_image = generator.render(image_tensor)

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
                synth_image = generator.render(synth_image_tensor)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def example_to_image(cfg, example: tf.train.Example, out_path: str, with_simulations=False, margin=10, max_replicates=1):
    features = {
        "image": _example_image(example),
    }
    _, replicates, *_ = _example_sim_images_shape(example)
    if with_simulations and replicates > 0:
        features["sim/images"] = _example_sim_images(example)
    
    features_to_image(cfg, features, out_path, with_simulations=with_simulations and replicates > 0, margin=margin, max_replicates=max_replicates)


def load_example_dataset(filenames, with_label=False, with_simulations=False) -> tf.data.Dataset:
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

    return tf.data.TFRecordDataset(filenames=filenames, compression_type=_filename_to_compression(filenames[0])).map(
        map_func=_process_input
    )

