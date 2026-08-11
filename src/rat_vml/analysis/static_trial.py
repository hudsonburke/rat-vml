"""Static trial selection and marker gap handling for scaling.

Provides functions to:
- Find static trials in Parquet data
- Filter to frames with complete named markers
- Write clean TRC files for OpenSim scaling
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Named markers used for scaling (not numeric markers like *0, *1)
NAMED_MARKERS = [
    "RTOE", "RHEE", "RANK", "RMKM", "RLML",
    "LTOE", "LHEE", "LANK", "LMKL", "LLML",
    "RASI", "LASI", "RPSI", "LPSI",
    "RKNM", "RKNL", "LKNM", "LKNL",
    "RTIB", "LTIB",
]


def find_static_trial(markers_df: pl.DataFrame) -> pl.DataFrame | None:
    """Find the static trial in a markers DataFrame.

    Prefers the last static trial (e.g. Static02 over Static01).
    Falls back to earlier trials if the last one has no clean frames.

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
    static_mask = pl.col("trial_name").str.to_lowercase().str.contains("static")
    static_trials = markers_df.filter(static_mask)

    if static_trials.is_empty():
        logger.warning("No static trial found")
        return None

    # Get static trial names, prefer last one
    trial_names = sorted(static_trials["trial_name"].unique().to_list(), reverse=True)

    for trial_name in trial_names:
        trial_df = markers_df.filter(pl.col("trial_name") == trial_name)

        # Filter to named markers only
        named_df = trial_df.filter(pl.col("marker_name").is_in(NAMED_MARKERS))

        # Check if we have at least one frame with all named markers present
        frames_with_all = named_df.group_by("frame").agg(
            pl.col("marker_name").n_unique().alias("n_markers")
        ).filter(pl.col("n_markers") >= len(NAMED_MARKERS) // 2)  # At least half the markers

        if not frames_with_all.is_empty():
            logger.info(f"Using static trial: {trial_name}")
            return trial_df

    logger.warning("No static trial with complete frames found")
    return None


def filter_complete_frames(
    markers_df: pl.DataFrame,
    min_markers: int = 10,
) -> pl.DataFrame:
    """Filter to frames where enough named markers are present.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame.
    min_markers : int
        Minimum number of named markers required per frame.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame with only complete frames.
    """
    # Filter to named markers only
    named_df = markers_df.filter(pl.col("marker_name").is_in(NAMED_MARKERS))

    # Count markers per frame
    frame_counts = named_df.group_by("frame").agg(
        pl.col("marker_name").n_unique().alias("n_markers")
    )

    # Keep frames with enough markers
    good_frames = frame_counts.filter(pl.col("n_markers") >= min_markers)["frame"]

    result = markers_df.filter(pl.col("frame").is_in(good_frames))
    n_removed = markers_df["frame"].n_unique() - result["frame"].n_unique()

    if n_removed > 0:
        logger.info(f"Removed {n_removed} frames with insufficient markers")

    return result


def find_clean_frame(markers_df: pl.DataFrame) -> int | None:
    """Find a single frame with complete named markers.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame.

    Returns
    -------
    int or None
        Frame number with complete markers, or None if not found.
    """
    # Filter to named markers
    named_df = markers_df.filter(pl.col("marker_name").is_in(NAMED_MARKERS))

    # Count markers per frame
    frame_counts = named_df.group_by("frame").agg(
        pl.col("marker_name").n_unique().alias("n_markers")
    )

    # Find frame with most markers
    best_frame = frame_counts.sort("n_markers", descending=True)

    if best_frame.is_empty():
        logger.warning("No clean frame found")
        return None

    frame = best_frame["frame"][0]
    n_markers = best_frame["n_markers"][0]
    logger.info(f"Found clean frame: {frame} ({n_markers} markers)")

    return frame


def prepare_static_trial_for_scaling(
    markers_df: pl.DataFrame,
    output_dir: Path,
    subject_id: str,
    session_id: str,
) -> Path | None:
    """Prepare a clean static trial TRC file for OpenSim scaling.

    Finds the static trial, filters to frames with complete markers,
    and writes a TRC file suitable for scaling.

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

    # Filter to frames with complete markers
    static_df = filter_complete_frames(static_df)

    if static_df.is_empty():
        logger.error("Static trial has no complete frames")
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
