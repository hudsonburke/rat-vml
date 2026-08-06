"""Parquet -> OpenSim file format extraction.

Reads marker and force plate data from Parquet files (written by
movedb-core) and writes TRC and MOT files for OpenSim IK/ID.

Uses osimpy's io module for file writing:
- osimpy.io.trc.df_to_trc + TRCMetadata
- osimpy.io.sto.df_to_sto + STOMetadata
- osimpy.io.forces.export_external_loads
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Vicon -> OpenSim coordinate rotation
_VICON_TO_OPENSIM = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)


def _read_markers(data_dir: Path, subject_id: str, session: str, trial_name: str) -> pl.DataFrame:
    path = data_dir / subject_id / "markers.parquet"
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(path)
    return df.filter(
        (pl.col("subject_id") == subject_id)
        & (pl.col("session_id") == session)
        & (pl.col("trial_name") == trial_name)
    )


def _read_forceplates(data_dir: Path, subject_id: str, session: str, trial_name: str) -> pl.DataFrame:
    path = data_dir / subject_id / "forceplates.parquet"
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(path)
    return df.filter(
        (pl.col("subject_id") == subject_id)
        & (pl.col("session_id") == session)
        & (pl.col("trial_name") == trial_name)
    )


def _detect_rate(df: pl.DataFrame) -> float:
    times = df["time"].unique().sort().to_list()
    if len(times) < 2:
        return 200.0
    dt = times[1] - times[0]
    return round(1.0 / dt, 1) if dt > 0 else 200.0


def _markers_to_wide_df(markers_df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Pivot markers from long to wide format for TRC writing.

    Returns (wide_df, marker_names) where wide_df has columns:
    Frame, Time, {name}_x, {name}_y, {name}_z for each marker.
    """
    marker_names = sorted(markers_df["marker_name"].unique().to_list())
    frames = markers_df.select(["time", "frame"]).unique().sort("frame")

    data = {"Frame": frames["frame"].to_list(), "Time": frames["time"].to_list()}

    for name in marker_names:
        m = markers_df.filter(pl.col("marker_name") == name).sort("frame")
        if len(m) < len(frames):
            padded = frames.join(m, on="frame", how="left").fill_null(0)
            x = padded["x"].to_numpy()
            y = padded["y"].to_numpy()
            z = padded["z"].to_numpy()
        else:
            x = m["x"].to_numpy()
            y = m["y"].to_numpy()
            z = m["z"].to_numpy()

        # Apply Vicon -> OpenSim rotation
        coords = np.column_stack([x, y, z]) @ _VICON_TO_OPENSIM.T
        data[f"{name}_x"] = coords[:, 0]
        data[f"{name}_y"] = coords[:, 1]
        data[f"{name}_z"] = coords[:, 2]

    return pl.DataFrame(data), marker_names


def _fp_to_wide_df(fp_df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Pivot force plates from long to wide format for MOT writing.

    Returns (wide_df, fp_names) where wide_df has columns:
    time, {fp}_force_x, {fp}_force_y, ..., {fp}_cop_x, ..., {fp}_moment_x, ...
    """
    fp_names = sorted(fp_df["fp_name"].unique().to_list())
    frames = fp_df.select(["time", "frame"]).unique().sort("frame")

    data = {"time": frames["time"].to_list()}

    for fp in fp_names:
        for var in ["force", "moment", "cop"]:
            for axis in ["x", "y", "z"]:
                col_name = f"{fp}_{var}_{axis}"
                vals = []
                for frame in frames["frame"].to_list():
                    v = fp_df.filter(
                        (pl.col("fp_name") == fp)
                        & (pl.col("variable") == var)
                        & (pl.col("axis") == axis)
                        & (pl.col("frame") == frame)
                    )
                    if v.is_empty():
                        vals.append(0.0)
                    else:
                        vals.append(float(v["value"].to_list()[0]))
                data[col_name] = vals

    return pl.DataFrame(data), fp_names


def parquet_to_trc(
    data_dir: str | Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: str | Path,
    output_name: str | None = None,
) -> Path:
    """Extract markers from Parquet and write TRC file using osimpy."""
    from osimpy.io.trc import df_to_trc, TRCMetadata

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = f"{trial_name}.trc"

    markers_df = _read_markers(data_dir, subject_id, session, trial_name)
    if markers_df.is_empty():
        raise ValueError(f"No marker data for {subject_id}/{session}/{trial_name}")

    wide_df, marker_names = _markers_to_wide_df(markers_df)
    rate = _detect_rate(markers_df)

    metadata = TRCMetadata(
        name=output_name,
        DataRate=rate,
        CameraRate=rate,
        NumFrames=len(wide_df),
        NumMarkers=len(marker_names),
        Units="mm",
        MarkerNames=marker_names,
    )

    output_path = output_dir / output_name
    df_to_trc(output_path, wide_df, metadata)
    logger.info(f"  Wrote TRC: {output_path} ({len(wide_df)} frames, {len(marker_names)} markers)")
    return output_path


def parquet_to_fp_mot(
    data_dir: str | Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: str | Path,
    output_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Extract force plate data from Parquet and write MOT + external loads XML using osimpy."""
    from osimpy.io.sto import df_to_sto, STOMetadata
    from osimpy.io.forces import export_external_loads, OpenSimExternalForce

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = trial_name

    fp_df = _read_forceplates(data_dir, subject_id, session, trial_name)
    if fp_df.is_empty():
        raise ValueError(f"No force plate data for {subject_id}/{session}/{trial_name}")

    wide_df, fp_names = _fp_to_wide_df(fp_df)
    rate = _detect_rate(fp_df)

    metadata = STOMetadata(
        name=f"{output_prefix}_forces.mot",
        nRows=len(wide_df),
        nColumns=len(wide_df.columns),
        inDegrees="yes",
    )

    mot_path = output_dir / f"{output_prefix}_forces.mot"
    df_to_sto(mot_path, wide_df, metadata)
    logger.info(f"  Wrote MOT: {mot_path} ({len(wide_df)} frames, {len(fp_names)} plates)")

    # Write external loads XML using osimpy
    ext_forces = []
    for fp in fp_names:
        ext_forces.append(OpenSimExternalForce(
            name=fp,
            applied_to_body="calcn_r",
            force_expressed_in_body="ground",
            point_expressed_in_body="ground",
            data_source_name=mot_path.name,
        ))

    ext_loads_path = output_dir / f"{output_prefix}_ext_loads.xml"
    export_external_loads(str(ext_loads_path), ext_forces, datafile_name=mot_path.name)
    logger.info(f"  Wrote ext loads: {ext_loads_path}")

    return mot_path, ext_loads_path
