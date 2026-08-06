#!/usr/bin/env python3
"""Aggregate results by group and session.

Usage::

    python scripts/aggregate.py --data-dir data/processed --group control --session week24
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results by group and session")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--group", required=True, help="Treatment group name")
    parser.add_argument("--session", required=True, help="Session/timepoint name")
    parser.add_argument("--result-type", default="ik", choices=["ik", "id", "moco"])
    parser.add_argument("--output-dir", "-o", default="data/aggregate", help="Output directory")
    args = parser.parse_args()

    from rat_vml.analysis.aggregation import load_results, aggregate_by_group

    results = load_results(args.data_dir, result_type=args.result_type, session=args.session)
    if results.is_empty():
        logger.error(f"No {args.result_type} results found for session {args.session}")
        return

    group_result = aggregate_by_group(results, group=args.group, session=args.session)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Group {args.group} at {args.session}: {group_result.n_subjects} subjects")
    logger.info(f"  Coordinates: {group_result.coord_names}")
    logger.info(f"  Mean shape: {group_result.mean.shape if group_result.mean is not None else 'N/A'}")


if __name__ == "__main__":
    main()
