"""Static trial selection and marker gap handling for scaling.

Provides functions to:
- Find static trials in Parquet data
- Detect and remove frames with marker gaps
- Write clean TRC files for OpenSim scaling
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def find_static_trial(markers_df: pl.DataFrame) -> pl.DataFrame | None:
    """Find the static trial in a markers DataFrame.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame with columns: frame, time, marker_name, x, y, z,
        subject_id, session_id, trial_name.

    Returns
    -------
    pl.DataFrame or None
        Filtered DataFrame for the static trial, or None if not found.
    """
    # Find trials with "static" in the name (case-insensitive)
    static_trials = markers_df.filter(
        pl.col("trial_name").str.to_lowercase().str.contains("static")
    )

    if static_trials.is_empty():
        logger.warning("No static trial found")
        return None

    # Get the first static trial
    trial_name = static_trials["trial_name"].unique()[0]
    logger.info(f"Found static trial: {trial_name}")

    return static_trials.filter(pl.col("trial_name") == trial_name)


def detect_marker_gaps(
    markers_df: pl.DataFrame,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Detect frames with marker gaps (all-zero positions).

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame with columns: frame, time, marker_name, x, y, z.
    threshold : float
        Values at or below this threshold are considered missing.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional column 'has_gap' (bool) indicating
        which frames have gaps.
    """
    # A frame has a gap if any marker has all coordinates at or below threshold
    marker_cols = ["x", "y", "z"]

    # Check if all coordinates are at or below threshold for each marker
    markers_df = markers_df.with_columns(
        ((pl.col("x").abs() <= threshold) &
         (pl.col("y").abs() <= threshold) &
         (pl.col("z").abs() <= threshold)).alias("marker_gap")
    )

    # Aggregate per frame: a frame has a gap if ANY marker has a gap
    frame_gaps = markers_df.group_by("frame").agg(
        pl.col("marker_gap").any().alias("has_gap")
    )

    # Join back to get has_gap per frame
    markers_df = markers_df.join(frame_gaps, on="frame", how="left")

    return markers_df


def remove_gap_frames(markers_df: pl.DataFrame) -> pl.DataFrame:
    """Remove frames with marker gaps.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame with 'has_gap' column from detect_marker_gaps().

    Returns
    -------
    pl.DataFrame
        DataFrame with gap frames removed.
    """
    if "has_gap" not in markers_df.columns:
        markers_df = detect_marker_gaps(markers_df)

    n_before = markers_df["frame"].n_unique()
    markers_df = markers_df.filter(~pl.col("has_gap"))
    n_after = markers_df["frame"].n_unique()

    n_removed = n_before - n_after
    if n_removed > 0:
        logger.info(f"Removed {n_removed} frames with marker gaps ({n_before} -> {n_after})")

    return markers_df.drop("has_gap", "marker_gap")


def find_clean_frame(markers_df: pl.DataFrame) -> int | None:
    """Find a single frame with no marker gaps.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame with columns: frame, marker_name, x, y, z.

    Returns
    -------
    int or None
        Frame number with no gaps, or None if all frames have gaps.
    """
    marker_cols = ["x", "y", "z"]

    # Check if any marker has all zeros at each frame
    has_gap = markers_df.group_by("frame").agg(
        ((pl.col("x").abs() <= 0.0) &
         (pl.col("y").abs() <= 0.0) &
         (pl.col("z").abs() <= 0.0)).any().alias("has_gap")
    )

    # Find first frame without gaps
    clean_frames = has_gap.filter(~pl.col("has_gap"))

    if clean_frames.is_empty():
        logger.warning("No clean frame found (all frames have gaps)")
        return None

    frame = clean_frames["frame"][0]
    logger.info(f"Found clean frame: {frame}")
    return frame


def prepare_static_trial_for_scaling(
    markers_df: pl.DataFrame,
    output_dir: Path,
    subject_id: str,
    session_id: str,
) -> Path | None:
    """Prepare a clean static trial TRC file for OpenSim scaling.

    Finds the static trial, removes frames with marker gaps, and writes
    a TRC file suitable for scaling.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Full markers DataFrame for the session.
    output_dir : Path
        Output directory for the TRC file.
    subject_id : str
        Subject identifier.
    session_id : str
        Session identifier.

    Returns
    -------
    Path or None
        Path to the written TRC file, or None if no static trial found.
    """
    from ..parquet_io import parquet_to_trc

    # Find static trial
    static_df = find_static_trial(markers_df)
    if static_df is None:
        return None

    # Remove frames with gaps
    static_df = remove_gap_frames(static_df)

    if static_df.is_empty():
        logger.error("Static trial has no clean frames after gap removal")
        return None

    # Find a clean frame for reference
    clean_frame = find_clean_frame(static_df)
    if clean_frame is None:
        logger.error("No clean frame found in static trial")
        return None

    # Write TRC file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trc_path = output_dir / f"{subject_id}_{session_id}_static.trc"

    # Pivot to wide format for TRC writing
    static_wide = static_df.filter(pl.col("frame") == clean_frame).pivot(
        on="marker_name",
        index="frame",
        values=["x", "y", "z"],
    )

    # Write using parquet_to_trc
    parquet_to_trc(static_wide, trc_path, frame_rate=200.0)

    logger.info(f"Wrote static trial TRC: {trc_path}")
    return trc_path
