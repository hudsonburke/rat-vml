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
from rat_vml.analysis.parquet_catalog import ParquetCatalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run IK for a subject/trial")
    parser.add_argument("--data-dir", required=True, help="Processed Parquet data directory")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--model", required=True, help="Scaled model path")
    parser.add_argument("--output-dir", "-o", default="data/results", help="Output directory")
    parser.add_argument("--cutoff", type=float, default=15.0, help="Filter cutoff frequency (Hz)")
    # IK-specific params
    parser.add_argument("--marker-weight", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--constraint-weight", type=float, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load markers and events
    markers_path = data_dir / args.subject / "markers.parquet"
    events_path = data_dir / args.subject / "events.parquet"

    markers_df = pl.read_parquet(markers_path).filter(pl.col("session_id") == args.session)

    # Validate walking trial
    cat = ParquetCatalog(data_dir)
    valid_df = cat.valid_walking_trials(min_events=7, session=args.session)
    valid_trials = valid_df.filter(pl.col("subject_id") == args.subject)["trial_name"].to_list()
    if args.trial not in valid_trials:
        logger.error(f"Trial {args.trial} is not a valid walking trial")
        logger.info(f"Valid trials: {valid_trials}")
        return

    # Filter markers
    trial_markers = markers_df.filter(pl.col("trial_name") == args.trial)
    filtered = filter_markers(trial_markers, cutoff_hz=args.cutoff)

    # Convert to wide TRC format
    wide = markers_to_wide_trc(filtered)

    # Write TRC file
    from osimpy.io.trc import df_to_trc, TRCMetadata

    frame_rate = 1.0 / (filtered["time"][1] - filtered["time"][0]) if len(filtered) > 1 else 200.0
    marker_names = [c.replace("_x", "").replace("_y", "").replace("_z", "")
                    for c in wide.columns if c.endswith("_x")]

    trc_path = output_dir / f"trc_{args.session}_{args.trial}.trc"
    metadata = TRCMetadata(
        name=trc_path.name,
        DataRate=frame_rate,
        CameraRate=frame_rate,
        NumFrames=len(wide),
        NumMarkers=len(marker_names),
        Units="mm",
        MarkerNames=marker_names,
    )
    df_to_trc(trc_path, wide, metadata)
    logger.info(f"Wrote TRC: {trc_path}")

    # Run IK
    from osimpy.tools import IKSettings

    ik_path = output_dir / f"ik_{args.session}_{args.trial}.mot"
    settings = IKSettings(
        model_path=Path(args.model),
        marker_path=trc_path,
        output_motion_file=str(ik_path),
        constraint_weight=args.constraint_weight,
        accuracy=args.accuracy,
    )
    result = settings.run()

    if result.success:
        logger.info(f"IK complete: {ik_path}")
    else:
        logger.error(f"IK failed: {result.errors}")


if __name__ == "__main__":
    main()
