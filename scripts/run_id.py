#!/usr/bin/env python3
"""Run ID for a subject/trial.

Usage::

    python scripts/run_id.py --subject BAA01 --trial Walk02 --model data/results/BAA01/scaled/model.osim --ik-file data/results/BAA01/trials/Walk02/BAA01_Walk02_ik.mot
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ID for a subject/trial")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--ik-file", required=True, help="IK result file")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output directory")
    args = parser.parse_args()

    from rat_vml.analysis.parquet_io import parquet_to_fp_mot
    from rat_vml.analysis.pipeline import run_id

    output_dir = Path(args.output_dir) / args.subject / "trials" / args.trial
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract force plate data from Parquet
    _, ext_loads_path = parquet_to_fp_mot(
        args.data_dir, args.subject, args.session, args.trial, output_dir
    )

    # Run ID
    id_file = run_id(Path(args.model), Path(args.ik_file), ext_loads_path, output_dir,
                     name=f"{args.subject}_{args.trial}")
    logger.info(f"ID complete: {id_file}")


if __name__ == "__main__":
    main()
