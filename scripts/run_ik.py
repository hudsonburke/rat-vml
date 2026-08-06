#!/usr/bin/env python3
"""Run IK for a subject/trial using osimpy directly.

Reads markers from Parquet, writes TRC using osimpy io, runs IK using osimpy IKSettings.

Usage::

    python scripts/run_ik.py --data-dir data/processed --subject BAA01 --session Baseline --trial Walk02 --model data/processed/BAA01/scaled_model.osim
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run IK for a subject/trial")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Output directory")
    # IK-specific params (from params.yaml via DVC)
    parser.add_argument("--marker-weight", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--constraint-weight", type=float, default=None)
    args = parser.parse_args()

    from osimpy.tools import IKSettings
    from rat_vml.analysis.parquet_io import parquet_to_trc

    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract TRC from Parquet
    trc_path = output_dir / f"trc_{args.session}_{args.trial}.trc"
    parquet_to_trc(args.data_dir, args.subject, args.session, args.trial, output_dir,
                   output_name=trc_path.name)

    # Run IK using osimpy
    ik_path = output_dir / f"ik_{args.session}_{args.trial}.mot"
    settings = IKSettings(
        model_path=Path(args.model),
        marker_path=trc_path,
        output_motion_file=str(ik_path),
        constraint_weight=args.constraint_weight,
        accuracy=args.accuracy,
    )
    result = settings.run()
    logger.info(f"IK complete: {ik_path}")


if __name__ == "__main__":
    main()
