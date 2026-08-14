#!/usr/bin/env python3
"""Scale model for a subject.

Reads anthropometrics from parameters.parquet, finds clean static trial,
filters markers, and scales the base model.

Usage::

    python scripts/scale.py --data-dir data/processed --subject BAA01 --session baseline --model models/rat_hindlimb_bilateral.osim
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from rat_vml.analysis.static_trial import (
    find_static_trial,
    find_clean_frame_range,
)
from rat_vml.analysis.filtering import filter_markers, markers_to_wide_trc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_session_params(data_dir: Path, subject_id: str, session_id: str) -> dict:
    """Load anthropometrics from parameters.parquet via movedb catalog."""
    from movedb import MoveDB

    db = MoveDB(data_dir)
    params_df = db.get_parameters(subject_id, session=session_id)

    if params_df.is_empty():
        raise ValueError(
            f"No parameters found for {subject_id}/{session_id}. "
            "Run convert.py to generate them."
        )

    row = params_df.to_dicts()[0]
    params = {
        "mass": row.get("Mass", 0.0),
        "r_femur_length": row.get("RFemurLength", 0.0),
        "r_tibia_length": row.get("RTibiaLength", 0.0),
        "r_foot_length": row.get("RFootLength", 0.0),
        "l_femur_length": row.get("LFemurLength", 0.0),
        "l_tibia_length": row.get("LTibiaLength", 0.0),
        "l_foot_length": row.get("LFootLength", 0.0),
    }

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
    parser.add_argument("--cutoff", type=float, default=15.0, help="Filter cutoff frequency (Hz)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load anthropometrics via movedb catalog
    params = load_session_params(data_dir, args.subject, args.session)

    # Load markers via movedb catalog
    from movedb import MoveDB
    db = MoveDB(data_dir)
    markers_df = db.get_points(args.subject, session=args.session)

    # Find static trial
    static_df = find_static_trial(markers_df)
    if static_df is None:
        logger.error("No static trial found")
        return

    # Find clean frame range
    frame_range = find_clean_frame_range(static_df)
    if frame_range is None:
        logger.error("No clean frame range in static trial")
        return

    start_frame, end_frame = frame_range
    logger.info(f"Using frame range {start_frame}-{end_frame} for scaling")

    frame_df = static_df.filter(
        (pl.col("frame") >= start_frame) & (pl.col("frame") <= end_frame)
    )

    # Filter markers
    filtered = filter_markers(frame_df, cutoff_hz=args.cutoff)

    # Convert to wide TRC format
    wide = markers_to_wide_trc(filtered)

    # Write TRC file
    from osimpy.io.trc import df_to_trc
    trc_path = output_dir / f"{args.subject}_{args.session}_static.trc"
    # Get frame rate from data
    time_vals = filtered.select(
        pl.col("time").filter(pl.col("frame") == filtered["frame"].min())
    )["time"].unique().sort().to_list()
    frame_rate = 1.0 / (time_vals[1] - time_vals[0]) if len(time_vals) > 1 else 200.0
    logger.info(f"Frame rate: {frame_rate} Hz")

    df_to_trc(wide, trc_path, frame_rate=frame_rate)
    logger.info(f"Wrote static trial TRC: {trc_path}")

    # Scale the model
    from rat_vml.analysis.scaling import scale_model_for_subject
    scaled_path = scale_model_for_subject(
        base_model=Path(args.model),
        subject_name=f"{args.subject}_{args.session}",
        mass=params["mass"],
        r_femur_length=params["r_femur_length"],
        r_tibia_length=params["r_tibia_length"],
        r_foot_length=params["r_foot_length"],
        l_femur_length=params["l_femur_length"],
        l_tibia_length=params["l_tibia_length"],
        l_foot_length=params["l_foot_length"],
        output_dir=output_dir,
    )

    logger.info(f"Scaled model: {scaled_path}")


if __name__ == "__main__":
    main()
