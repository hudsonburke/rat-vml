"""Static trial selection for scaling.

Finds a static trial with all required markers present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Required markers for scaling
REQUIRED_MARKERS = [
    "TAIL", "SPL6", "LASI", "RASI",
    "LHIP", "LKNE", "LANK", "LTOE",
    "RHIP", "RKNE", "RANK", "RTOE",
]


def find_static_trial(markers_df: pl.DataFrame) -> pl.DataFrame | None:
    """Find the last static trial with all required markers.

    Prefers Static02 over Static01. Returns None if no static trial
    has all required markers.
    """
    # Filter to static trials
    static_mask = pl.col("trial_name").str.to_lowercase().str.contains("static")
    static_df = markers_df.filter(static_mask)

    if static_df.is_empty():
        logger.warning("No static trial found")
        return None

    # Get static trial names, prefer last one
    trial_names = sorted(static_df["trial_name"].unique().to_list(), reverse=True)

    for trial_name in trial_names:
        trial_df = markers_df.filter(pl.col("trial_name") == trial_name)

        # Check if this trial has all required markers
        present = set(trial_df["marker_name"].unique().to_list())
        missing = set(REQUIRED_MARKERS) - present

        if not missing:
            logger.info(f"Using static trial: {trial_name}")
            return trial_df

        logger.debug(f"  {trial_name}: missing {missing}")

    logger.warning("No static trial with all required markers found")
    return None


def find_clean_frame(markers_df: pl.DataFrame) -> int | None:
    """Find a frame where all required markers are present.

    Returns the first such frame, or None.
    """
    # Filter to required markers only
    required_df = markers_df.filter(pl.col("marker_name").is_in(REQUIRED_MARKERS))

    # Count required markers per frame
    frame_counts = required_df.group_by("frame").agg(
        pl.col("marker_name").n_unique().alias("n_markers")
    )

    # Find frames with all required markers
    complete = frame_counts.filter(pl.col("n_markers") >= len(REQUIRED_MARKERS))

    if complete.is_empty():
        logger.warning("No frame with all required markers")
        return None

    frame = complete["frame"][0]
    logger.info(f"Found clean frame: {frame}")
    return frame


def prepare_static_trial_for_scaling(
    markers_df: pl.DataFrame,
    output_dir: Path,
    subject_id: str,
    session_id: str,
) -> Path | None:
    """Prepare a clean static trial TRC for OpenSim scaling.

    Finds the last static trial with all required markers,
    picks a clean frame, and writes a TRC file.
    """
    from ..parquet_io import parquet_to_trc

    static_df = find_static_trial(markers_df)
    if static_df is None:
        return None

    clean_frame = find_clean_frame(static_df)
    if clean_frame is None:
        logger.error("No clean frame in static trial")
        return None

    # Write TRC
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trc_path = output_dir / f"{subject_id}_{session_id}_static.trc"

    frame_df = static_df.filter(pl.col("frame") == clean_frame).pivot(
        on="marker_name",
        index="frame",
        values=["x", "y", "z"],
    )

    parquet_to_trc(frame_df, trc_path, frame_rate=200.0)
    logger.info(f"Wrote static trial TRC: {trc_path}")
    return trc_path
