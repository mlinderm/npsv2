import argparse, logging
from ..simulation import filter_reads_gc, filter_reads_gnomad
from ..multiallelic import filter_nonref, merge_into_multiallelic
from ..propose.refine import train_model

def main():
    parser = argparse.ArgumentParser(
        "npsv2u", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    logging_options = parser.add_mutually_exclusive_group()
    logging_options.add_argument(
        "-d",
        "--debug",
        help="Debug logging",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    logging_options.add_argument(
        "-v",
        "--verbose",
        help="Verbose logging",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    parser_gccovg = subparsers.add_parser(
        "gc_covg", help="Filter synthetic reads based on gc coverage profile"
    )
    parser_gccovg.add_argument(
        "--fasta-path",
        dest="fasta_path",
        type=str,
        help="Path to FASTA file used to generate the synthetic reads",
        required=True,
    )
    parser_gccovg.add_argument(
        "--stats-path",
        dest="stats_path",
        type=str,
        help="Path to stats JSON file generated by preprocessing command",
        required=True,
    )
    parser_gccovg.add_argument(
        "-i", "--input", help="Input SAM file.", type=str, dest="input", required=True
    )

    parser_gnomadcovg = subparsers.add_parser(
        "gnomad_covg", help="Filter synthetic reads based on gnomAD coverage profile"
    )
    parser_gnomadcovg.add_argument(
        "--covg-path",
        dest="covg_path",
        type=str,
        help="Path to file with gnomAD coverage profile",
        required=True,
    )
    parser_gnomadcovg.add_argument(
        "-i", "--input", help="Input SAM file.", type=str, dest="input", required=True
    )

    parser_filter = subparsers.add_parser(
        "filter_nonref", help="Filter VCF to just non-reference genotypes in SV regions"
    )
    parser_filter.add_argument(
        "-i", "--input", help="Input VCF file.", type=str, required=True
    )
    parser_filter.add_argument(
        "-o", "--output", help="Output VCF file.", default="/dev/stdout"
    )
    parser_filter.add_argument(
        "-R", "--reference", help="Reference fasta.", type=str, required=True,
    )
    parser_filter.add_argument(
        "--sample", help="Sample genotypes to use in filter", type=str, required=True
    )
    parser_filter.add_argument(
        "--flank", help="Flank (bp) for determining variant overlap", default=1000
    )
    parser_filter.add_argument(
        "--keep_noref", help="Keep records containing or overlapping 'N's", action="store_true"
    )
    
    parser_refinetrain = subparsers.add_parser(
        "refine_train", help="Train model for refine with RF"
    )
    parser_refinetrain.add_argument(
        "-i", "--input", help="Input VCF file.", type=str, required=True
    )
    parser_refinetrain.add_argument(
        "-o", "--output", help="Specify output file", type=str, required=True
    )
    parser_refinetrain.add_argument(
        "-p", "--pbsv", help="Input PBSV file.", type=str, required=True
    )

    parser_multiallelic = subparsers.add_parser(
        "multiallelic", help="Merge biallelic SVs into multiallelic SVs"
    )
    parser_multiallelic.add_argument(
        "-i", "--input", help="Input VCF file.", type=str, required=True
    )
    parser_multiallelic.add_argument(
        "-o", "--output", help="Output VCF file.", default="/dev/stdout"
    )
    parser_multiallelic.add_argument(
        "-R", "--reference", help="Reference fasta.", type=str, required=True,
    )
    parser_multiallelic.add_argument(
        "--flank", help="Expand variants by flank during merging", default=0,
    )


    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    if args.command == "gc_covg":
        filter_reads_gc(args.stats_path, args.fasta_path, args.input, "/dev/stdout")
    elif args.command == "gnomad_covg":
        filter_reads_gnomad(args.covg_path, args.input, "/dev/stdout")
    elif args.command == "filter_nonref":
        filter_nonref(args.reference, args.input, args.output, args.sample, flank=args.flank, drop_noref=(not args.keep_noref))
    elif args.command == "refine_train":
        train_model(args.input, args.pbsv, args.output)
    elif args.command == "multiallelic":
        merge_into_multiallelic(args.input, args.output, args.reference, flank=args.flank)

if __name__ == "__main__":
    main()