import argparse
from ..simulation import filter_reads_gc, filter_reads_gnomad
from ..multiallelic import filter_nonref
from ..propose.refine import train_model

def main():
    parser = argparse.ArgumentParser(
        "npsv2u", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--sample", help="Sample genotypes to use in filter", type=str, required=True
    )
    parser_filter.add_argument(
        "--flank", help="Flank (bp) for determining variant overlap", default=500
    )
    
    parser_refinetrain = subparsers.add_parser(
        "refine_train", help="train model for refine with RF"
    )
    parser_refinetrain.add_argument(
        "-i", "--input", help="Input VCF file.", type=str, required=True
    )
    parser_refinetrain.add_argument(
        "-o", "--output", help="Specify directory to store output files", type=str, required=True
    )
    parser_refinetrain.add_argument(
        "-p", "--pbsv", help="Input PBSV file.", type=str, required=True
    )

    args = parser.parse_args()

    if args.command == "gc_covg":
        filter_reads_gc(args.stats_path, args.fasta_path, args.input, "/dev/stdout")
    elif args.command == "gnomad_covg":
        filter_reads_gnomad(args.covg_path, args.input, "/dev/stdout")
    elif args.command == "filter_nonref":
        filter_nonref(args.input, args.output, args.sample, flank=args.flank)
    elif args.command == "refine_train":
        train_model(args.input, args.pbsv, args.output)

if __name__ == "__main__":
    main()