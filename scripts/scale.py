#!/usr/bin/env python3
"""Scale model for a subject.

Reads anthropometrics from sessions.parquet and scales the base model.

Usage::

    python scripts/scale.py --data-dir data/processed --subject BAA01 --session baseline --model models/rat_hindlimb_bilateral.osim
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from rat_vml.analysis.pipeline import scale_model_for_subject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_session_params(data_dir: Path, subject_id: str, session_id: str) -> dict:
    """Load anthropometrics from sessions.parquet.

    Parameters
    ----------
    data_dir : Path
        Processed data directory (contains subject subdirectories).
    subject_id : str
        Subject identifier (e.g. "BAA01").
    session_id : str
        Session name (e.g. "baseline", "week24").

    Returns
    -------
    dict
        Dictionary with keys: mass_kg, rfemur_length_mm, rtibia_length_mm,
        lfemur_length_mm, ltibia_length_mm, rfoot_length_mm, lfoot_length_mm.
    """
    sessions_path = data_dir / subject_id / "sessions.parquet"
    if not sessions_path.exists():
        raise FileNotFoundError(
            f"sessions.parquet not found for {subject_id}. "
            f"Run convert.py to generate it."
        )

    df = pl.read_parquet(sessions_path)
    session_row = df.filter(pl.col("session_id") == session_id)

    if session_row.is_empty():
        available = df["session_id"].unique().to_list()
        raise ValueError(
            f"Session '{session_id}' not found for {subject_id}. "
            f"Available: {available}"
        )

    row = session_row.to_dicts()[0]

    # Map C3D PROCESSING parameter names to our function args
    params = {
        "mass": row.get("Mass", 0.0),
        "r_femur_length": row.get("RFemurLength", 0.0),
        "r_tibia_length": row.get("RTibiaLength", 0.0),
        "r_foot_length": row.get("RFootLength", 0.0),
        "l_femur_length": row.get("LFemurLength", 0.0),
        "l_tibia_length": row.get("LTibiaLength", 0.0),
        "l_foot_length": row.get("LFootLength", 0.0),
    }

    # Log extracted parameters
    logger.info(f"Session parameters for {subject_id}/{session_id}:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")

    return params


def main():
    parser = argparse.ArgumentParser(description="Scale model for a subject")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name (e.g. baseline, week24)")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load anthropometrics from sessions.parquet
    params = load_session_params(data_dir, args.subject, args.session)

    # Scale the model
    scaled_path = scale_model_for_subject(
        base_model=Path(args.model),
        subject_name=f"{args.subject}_{args.session}",
        mass=params["mass"],
        r_femur_length=params["r_femur_length"],
        r_tibia_length=params["r_tibia_length"],
        output_dir=output_dir,
        r_foot_length=params["r_foot_length"],
        l_femur_length=params["l_femur_length"],
        l_tibia_length=params["l_tibia_length"],
        l_foot_length=params["l_foot_length"],
    )

    logger.info(f"Scaled model: {scaled_path}")


if __name__ == "__main__":
    main()
