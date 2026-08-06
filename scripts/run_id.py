#!/usr/bin/env python3
"""Run ID for a subject/trial using osimpy directly.

Reads force plates from Parquet, writes MOT + ext loads using osimpy io, runs ID using osimpy IDSettings.

Usage::

    python scripts/run_id.py --data-dir data/processed --subject BAA01 --session Baseline --trial Walk02 --model data/processed/BAA01/scaled_model.osim --ik-file data/processed/BAA01/ik_Baseline_Walk02.mot
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
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Output directory")
    # ID-specific params (from params.yaml via DVC)
    parser.add_argument("--lowpass-cutoff-frequency", type=float, default=6.0)
    args = parser.parse_args()

    from osimpy.tools import IDSettings
    from rat_vml.analysis.parquet_io import parquet_to_fp_mot

    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract force plate data from Parquet
    _, ext_loads_path = parquet_to_fp_mot(
        args.data_dir, args.subject, args.session, args.trial, output_dir,
        output_prefix=f"{args.session}_{args.trial}"
    )

    # Run ID using osimpy
    id_path = output_dir / f"id_{args.session}_{args.trial}.sto"
    settings = IDSettings(
        model_path=Path(args.model),
        coordinates_path=Path(args.ik_file),
        output_forces_file=str(id_path),
        external_loads_path=ext_loads_path,
        lowpass_cutoff_frequency=args.lowpass_cutoff_frequency,
    )
    result = settings.run()
    logger.info(f"ID complete: {id_path}")


if __name__ == "__main__":
    main()
