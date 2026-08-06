#!/usr/bin/env python3
"""Run MocoInverse for a subject/trial.

Usage::

    python scripts/run_moco.py --model data/results/BAA01/scaled/model.osim --ik-file data/results/BAA01/trials/Walk02/BAA01_Walk02_ik.mot --ext-loads data/results/BAA01/trials/Walk02/Walk02_ext_loads.xml
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MocoInverse for a subject/trial")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--ik-file", required=True, help="IK result file")
    parser.add_argument("--ext-loads", required=True, help="External loads XML")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output directory")
    args = parser.parse_args()

    from rat_vml.analysis.pipeline import run_moco

    output_dir = Path(args.output_dir) / args.subject / "trials" / args.trial
    output_dir.mkdir(parents=True, exist_ok=True)

    moco_file = run_moco(
        model_path=Path(args.model),
        coordinates_path=Path(args.ik_file),
        external_loads_path=Path(args.ext_loads),
        output_dir=output_dir,
        name=f"{args.subject}_{args.trial}_moco",
    )
    logger.info(f"MocoInverse complete: {moco_file}")


if __name__ == "__main__":
    main()
