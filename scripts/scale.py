#!/usr/bin/env python3
"""Scale model for a subject.

Usage::

    python scripts/scale.py --data-dir data/processed --subject BAA01 --session Baseline --model models/rat_hindlimb_bilateral.osim
"""

import argparse
import logging
from pathlib import Path

from rat_vml.analysis.pipeline import scale_model_for_subject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scale model for a subject")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output directory")
    args = parser.parse_args()

    # Load subject metadata from Parquet
    from rat_vml.analysis.parquet_catalog import ParquetCatalog
    cat = ParquetCatalog(args.data_dir)

    # TODO: load anthropometrics from subjects.csv or Parquet metadata
    # For now, require them as arguments or from a config file
    logger.info(f"Scaling model for {args.subject}/{args.session}")
    logger.info("TODO: load anthropometrics and call scale_model_for_subject()")


if __name__ == "__main__":
    main()
