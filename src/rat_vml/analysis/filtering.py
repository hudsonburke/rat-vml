"""Marker filtering for IK/ID.

Applies Butterworth lowpass filter to marker data.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_HZ = 15.0
DEFAULT_ORDER = 4


def butterworth_filter(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth lowpass filter (filtfilt)."""
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, data, axis=0)


def filter_markers(
    markers_df: pl.DataFrame,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> pl.DataFrame:
    """Filter marker positions with Butterworth lowpass.

    Works on long-format markers (frame, time, marker_name, x, y, z).
    Filters each marker's x, y, z jointly.
    """
    # Calculate frame rate from time spacing
    frames = markers_df["frame"].unique().sort()
    if len(frames) < 2:
        return markers_df

    time_vals = markers_df.filter(
        pl.col("frame") == frames[0]
    )["time"].to_list()

    if len(time_vals) < 2:
        return markers_df

    frame_rate = 1.0 / (time_vals[1] - time_vals[0])

    # Group by marker and apply filter to each
    def _filter_group(group_df: pl.DataFrame) -> pl.DataFrame:
        xyz = group_df.select(["x", "y", "z"]).to_numpy()
        filtered = butterworth_filter(xyz, cutoff_hz, frame_rate, order)
        return group_df.with_columns([
            pl.Series("x", filtered[:, 0]),
            pl.Series("y", filtered[:, 1]),
            pl.Series("z", filtered[:, 2]),
        ])

    result = markers_df.group_by("marker_name", maintain_order=True).map_groups(
        lambda g: _filter_group(g)
    )

    logger.info(f"Filtered markers at {cutoff_hz} Hz")
    return result


def markers_to_wide_trc(markers_df: pl.DataFrame) -> pl.DataFrame:
    """Convert long-format markers to wide-format TRC layout."""
    return markers_df.pivot(
        on="marker_name",
        index=["frame", "time"],
        values=["x", "y", "z"],
    )
