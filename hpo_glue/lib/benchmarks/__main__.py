from __future__ import annotations

import argparse
import logging

from hpo_glue.lib.benchmarks import BENCHMARKS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main(
    list: bool
):
    if list:
        logger.info("Available benchmarks:")
        for name in BENCHMARKS:
            logger.info(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available benchmarks",
    )
    args = parser.parse_args()
    main(args.list)
