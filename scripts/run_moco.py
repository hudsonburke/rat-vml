#!/usr/bin/env python3
"""Run MocoInverse for a subject/trial using osimpy directly.

Usage::

    python scripts/run_moco.py --subject BAA01 --session Baseline --trial Walk02 --model data/processed/BAA01/scaled_model.osim --ik-file data/processed/BAA01/ik_Baseline_Walk02.mot --ext-loads data/processed/BAA01/ext_loads_Baseline_Walk02.xml
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MocoInverse for a subject/trial")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--ik-file", required=True, help="IK result file")
    parser.add_argument("--ext-loads", required=True, help="External loads XML")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Output directory")
    # Moco params (from params.yaml via DVC)
    parser.add_argument("--mesh-interval", type=float, default=0.02)
    parser.add_argument("--replace-muscles-with-dgf", type=bool, default=True)
    parser.add_argument("--reserve-optimal-force", type=float, default=1.0)
    args = parser.parse_args()

    from osimpy.moco.inverse import MocoInverseSettings

    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run MocoInverse using osimpy
    moco_path = output_dir / f"moco_{args.session}_{args.trial}.sto"
    settings = MocoInverseSettings(
        model_path=Path(args.model),
        coordinates_path=Path(args.ik_file),
        external_loads_path=Path(args.ext_loads),
        results_directory=output_dir,
        solution_filename=f"moco_{args.session}_{args.trial}.sto",
        mesh_interval=args.mesh_interval,
        replace_muscles_with_dgf=args.replace_muscles_with_dgf,
        reserve_optimal_force=args.reserve_optimal_force,
    )
    result = settings.run()
    logger.info(f"MocoInverse complete: {moco_path}")


if __name__ == "__main__":
    main()
