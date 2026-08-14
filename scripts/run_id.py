#!/usr/bin/env python3
"""Run ID for a subject/trial.

Pipeline:
1. Filter force plates (58-62 Hz notch + 50 Hz lowpass)
2. Zero outside gait cycle
3. Export to MOT + ext loads XML
4. Run ID using scaled model + IK results

Usage::

    python scripts/run_id.py --data-dir data/processed --subject BAA01 --session baseline --trial Walk02 --model data/results/BAA01_scaled.osim --ik-file data/results/BAA01/ik_baseline_Walk02.mot
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from rat_vml.analysis.filtering import (
    filter_forceplate,
    zero_outside_gait_cycle,
)
from rat_vml.analysis.parquet_io import _fp_to_wide_df, _detect_rate

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
    parser.add_argument("--lowpass-cutoff", type=float, default=6.0, help="ID lowpass cutoff")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use movedb catalog
    from movedb import MoveDB
    db = MoveDB(data_dir)

    # Load force plates and events
    fp_df = db.get_forceplates(args.subject, session=args.session)
    fp_df = fp_df.filter(pl.col("trial_name") == args.trial)

    events_df = db.get_events(args.subject, session=args.session)
    events_df = events_df.filter(pl.col("trial_name") == args.trial)

    if fp_df.is_empty():
        logger.error(f"No force plate data for {args.subject}/{args.session}/{args.trial}")
        return

    # Filter force plates (58-62 Hz notch + 50 Hz lowpass)
    filtered = filter_forceplate(fp_df)

    # Zero outside gait cycle
    filtered = zero_outside_gait_cycle(filtered, events_df)

    # Convert to wide format and write MOT
    from osimpy.io.sto import df_to_sto, STOMetadata
    from osimpy.io.forces import export_external_loads, OpenSimExternalForce

    wide_df, fp_names = _fp_to_wide_df(filtered)
    rate = _detect_rate(filtered)

    output_prefix = f"{args.session}_{args.trial}"
    mot_path = output_dir / f"{output_prefix}_forces.mot"

    metadata = STOMetadata(
        name=mot_path.name,
        nRows=len(wide_df),
        nColumns=len(wide_df.columns),
        inDegrees="yes",
    )
    df_to_sto(mot_path, wide_df, metadata)
    logger.info(f"Wrote MOT: {mot_path}")

    # Write external loads XML
    # Determine body from force plate side
    side_col = "side" if "side" in filtered.columns else None
    ext_forces = []
    for fp in fp_names:
        if side_col:
            side = filtered.filter(pl.col("fp_name") == fp)[side_col][0]
            body = "foot_r" if side.lower() == "right" else "foot_l"
        else:
            body = "foot_r"
        ext_forces.append(OpenSimExternalForce(
            name=fp,
            applied_to_body=body,
            force_expressed_in_body="ground",
            point_expressed_in_body="ground",
            data_source_name=mot_path.name,
        ))

    ext_loads_path = output_dir / f"{output_prefix}_ext_loads.xml"
    export_external_loads(str(ext_loads_path), ext_forces, datafile_name=mot_path.name)
    logger.info(f"Wrote ext loads: {ext_loads_path}")

    # Run ID
    from osimpy.tools import IDSettings

    id_path = output_dir / f"id_{args.session}_{args.trial}.sto"
    settings = IDSettings(
        model_path=Path(args.model),
        coordinates_path=Path(args.ik_file),
        output_forces_file=str(id_path),
        external_loads_path=ext_loads_path,
        lowpass_cutoff_frequency=args.lowpass_cutoff,
    )
    result = settings.run()

    if result.success:
        logger.info(f"ID complete: {id_path}")
    else:
        logger.error(f"ID failed: {result.errors}")


if __name__ == "__main__":
    main()
