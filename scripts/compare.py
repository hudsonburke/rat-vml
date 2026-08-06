#!/usr/bin/env python3
"""Compare two groups at a given session/timepoint.

Usage::

    python scripts/compare.py --data-dir data/processed --group-a control --group-b temr --session week24
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare two groups")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--group-a", required=True, help="First group name")
    parser.add_argument("--group-b", required=True, help="Second group name")
    parser.add_argument("--session", required=True, help="Session/timepoint name")
    parser.add_argument("--result-type", default="ik", choices=["ik", "id", "moco"])
    parser.add_argument("--output-dir", "-o", default="data/comparisons", help="Output directory")
    args = parser.parse_args()

    from rat_vml.analysis.aggregation import load_results, aggregate_by_group, compare_groups

    results = load_results(args.data_dir, result_type=args.result_type, session=args.session)
    if results.is_empty():
        logger.error(f"No {args.result_type} results found for session {args.session}")
        return

    group_a = aggregate_by_group(results, group=args.group_a, session=args.session)
    group_b = aggregate_by_group(results, group=args.group_b, session=args.session)

    if group_a.n_subjects == 0 or group_b.n_subjects == 0:
        logger.error("One or both groups have no subjects")
        return

    comparison = compare_groups(group_a, group_b)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Comparison: {args.group_a} (n={group_a.n_subjects}) vs {args.group_b} (n={group_b.n_subjects})")
    logger.info(f"Session: {args.session}")
    logger.info(f"Coordinates: {comparison.coord_names}")

    # Log SPM results
    for coord, spm_result in comparison.spm_results.items():
        if spm_result is not None:
            logger.info(f"  {coord}: {spm_result}")


if __name__ == "__main__":
    main()
