"""Unified ANDES command dispatcher."""

from __future__ import annotations

import argparse
import sys

from . import compare, enrich, index_cli, null_cli

COMMANDS = {
    "compare": compare.main,
    "enrich": enrich.main,
    "index": index_cli.main,
    "null": null_cli.main,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="andes",
        description="ANDES gene-set comparison and ranked enrichment",
    )
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("compare", add_help=False, help="compare two GMT databases")
    subcommands.add_parser("enrich", add_help=False, help="score a ranked gene list")
    subcommands.add_parser(
        "index", add_help=False, help="build or query persistent indexes"
    )
    subcommands.add_parser(
        "null", add_help=False, help="build, list, or verify null artifacts"
    )
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    command = arguments[0]
    try:
        handler = COMMANDS[command]
    except KeyError:
        build_parser().error(f"unknown command {command!r}")
    return int(handler(arguments[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
