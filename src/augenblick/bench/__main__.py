"""Command-line operations for benchmark trees."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from augenblick.bench import audit, links, recon
from augenblick.bench.tree import Tree
from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)

OPERATIONS = {"links": links, "audit": audit, "recon": recon}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augenblick.bench", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tree", type=Path, required=True,
                        help="the experiment directory to operate on")
    common.add_argument("--only", action="append", help="limit to this specimen")
    common.add_argument("--index", type=int,
                        help="handle one specimen by its position in the roster")
    common.add_argument("--list", dest="listing", help="a specimen list to index into")

    operations = parser.add_subparsers(dest="operation", required=True,
                                       metavar="operation")
    for name, module in OPERATIONS.items():
        operation = operations.add_parser(name, parents=[common],
                                          help=module.__doc__.splitlines()[0])
        module.add_arguments(operation)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return OPERATIONS[args.operation].run(Tree(args.tree), args)
    except SceneError as error:
        logger.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
