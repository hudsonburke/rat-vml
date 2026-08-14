#!/usr/bin/env python3
"""Run IK for a subject/trial.

Pipeline:
1. Find valid walking trials (7 events, correct order)
2. Filter markers (15 Hz Butterworth)
3. Export to TRC
4. Run IK using scaled model

Usage::

    python scripts/run_ik.py --data-dir data/processed --subject BAA01 --session baseline --trial Walk02 --model data/results/BAA01_scaled.osim
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from rat_vml.analysis.filtering import filter_markers, markers_to_wide_trc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run IK for a subject/trial")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output base directory (results go to {output-dir}/{subject}/{session}/)")
    parser.add_argument("--cutoff", type=float, default=15.0, help="Filter cutoff frequency (Hz)")
    # IK-specific params
    parser.add_argument("--marker-weight", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--constraint-weight", type=float, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.subject / args.session
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use movedb catalog
    from movedb import MoveDB
    db = MoveDB(data_dir)

    # Load markers and events
    markers_df = db.get_points(args.subject, session=args.session)
    markers_df = markers_df.filter(pl.col("trial_name") == args.trial)

    # Validate walking trial
    events_df = db.get_events(args.subject, session=args.session)
    events_df = events_df.filter(pl.col("trial_name") == args.trial)

    if events_df.is_empty() or len(events_df) < 7:
        logger.error(f"Trial {args.trial} is not a valid walking trial (need 7+ events)")
        return

    # Determine time range from events
    # Events seem to have a time offset - find the range where markers have data
    first_foot_strike = events_df.filter(
        (pl.col("label") == "Foot Strike") & (pl.col("context") == "Right")
    )["time"].min()
    last_foot_strike = events_df.filter(
        (pl.col("label") == "Foot Strike") & (pl.col("context") == "Left")
    )["time"].max()
    
    # Find where we actually have valid marker data
    valid_data = markers_df.filter(pl.col("x").is_not_nan())
    if valid_data.is_empty():
        logger.error("No valid marker data found")
        return
    
    valid_time_min = valid_data["time"].min()
    valid_time_max = valid_data["time"].max()
    
    # Check if events align with marker data
    if first_foot_strike < valid_time_min or last_foot_strike > valid_time_max:
        # Events don't align with marker data - use marker data range
        logger.warning(f"Events ({first_foot_strike:.3f}-{last_foot_strike:.3f}s) "
                       f"don't align with marker data ({valid_time_min:.3f}-{valid_time_max:.3f}s)")
        time_range = (valid_time_min, valid_time_max)
    else:
        time_range = (first_foot_strike, last_foot_strike)
    
    logger.info(f"Using time range: {time_range[0]:.3f} - {time_range[1]:.3f} s")

    # Filter markers
    filtered = filter_markers(markers_df, cutoff_hz=args.cutoff, time_range=time_range)

    # Convert to wide TRC format
    wide = markers_to_wide_trc(filtered)

    # Write TRC file
    from osimpy.io.trc import df_to_trc, TRCMetadata
    from rat_vml.defaults import VICON_TO_OPENSIM

    # Get frame rate from unique times at a single frame
    time_vals = filtered.select(
        pl.col("time").filter(pl.col("frame") == filtered["frame"].min())
    )["time"].unique().sort().to_list()
    frame_rate = 1.0 / (time_vals[1] - time_vals[0]) if len(time_vals) > 1 else 200.0
    # Extract marker names from columns like 'TAIL_x', 'TAIL_y', 'TAIL_z'
    marker_cols = [c for c in wide.columns if c not in ("frame", "time")]
    all_marker_names = sorted(set(col.rsplit("_", 1)[0] for col in marker_cols))

    from rat_vml.defaults import REQUIRED_MARKERS
    required_markers = set(REQUIRED_MARKERS)
    
    # Filter to required markers that exist in the data
    marker_names = [m for m in all_marker_names if m in required_markers]
    logger.info(f"Using {len(marker_names)} of {len(all_marker_names)} markers for IK")

    # Filter wide dataframe to only include valid markers
    cols_to_keep = ["frame", "time"]
    for m in marker_names:
        for suffix in ["_x", "_y", "_z"]:
            col = f"{m}{suffix}"
            if col in wide.columns:
                cols_to_keep.append(col)
    wide = wide.select(cols_to_keep)
    
    logger.info(f"TRC has {wide.shape[0]} frames")

    trc_path = output_dir / f"{args.subject}_{args.session}_{args.trial}.trc"
    metadata = TRCMetadata(
        name=trc_path.name,
        DataRate=frame_rate,
        CameraRate=frame_rate,
        NumFrames=len(wide),
        NumMarkers=len(marker_names),
        Units="mm",
        MarkerNames=marker_names,
    )
    # Apply Vicon -> OpenSim rotation when writing TRC
    df_to_trc(trc_path, wide, metadata, rotation=VICON_TO_OPENSIM)
    logger.info(f"Wrote TRC: {trc_path}")

    # Run IK
    from osimpy.tools import IKSettings
    import rathindlimb

    ik_filename = f"{args.subject}_{args.session}_{args.trial}_ik.mot"
    settings = IKSettings(
        name=f"{args.subject}_{args.session}_{args.trial}",
        setup_path=rathindlimb.bilateral_ik_setup(),
        model_path=Path(args.model),
        marker_path=trc_path,
        results_directory=output_dir,
        output_motion_file=ik_filename,
        constraint_weight=args.constraint_weight,
        accuracy=args.accuracy,
    )
    result = settings.run()

    ik_path = output_dir / ik_filename
    if result.success:
        # Rename marker error and location files to consistent naming
        for old_name, suffix in [
            ("_ik_marker_errors.sto", "_ik_marker_errors.sto"),
            ("_ik_model_marker_locations.sto", "_ik_model_marker_locations.sto"),
        ]:
            old_path = output_dir / old_name
            if old_path.exists():
                new_name = f"{args.subject}_{args.session}_{args.trial}{suffix}"
                new_path = output_dir / new_name
                old_path.rename(new_path)
                logger.info(f"Renamed {old_name} -> {new_name}")
        logger.info(f"IK complete: {ik_path}")
    else:
        logger.error(f"IK failed: {result.errors}")


if __name__ == "__main__":
    main()
