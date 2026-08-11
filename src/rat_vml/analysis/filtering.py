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


# =========================================================================
# Force plate filtering
# =========================================================================

# Default filter parameters for force plates (matching MATLAB pipeline)
DEFAULT_FP_LOWPASS_HZ = 50.0
DEFAULT_FP_LOWPASS_ORDER = 4
NOTCH_CENTER_HZ = 60.0
NOTCH_WIDTH_HZ = 4.0  # 58-62 Hz notch


def notch_filter(
    data: np.ndarray,
    center_hz: float,
    width_hz: float,
    sample_rate: float,
    order: int = 2,
) -> np.ndarray:
    """Apply notch filter to remove powerline interference.

    Parameters
    ----------
    data : np.ndarray
        Signal to filter.
    center_hz : float
        Center frequency of notch (e.g., 60 Hz).
    width_hz : float
        Width of notch (e.g., 4 Hz for 58-62 Hz).
    sample_rate : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    from scipy.signal import iirnotch

    nyquist = sample_rate / 2.0
    low = (center_hz - width_hz / 2) / nyquist
    high = (center_hz + width_hz / 2) / nyquist

    # Band-stop filter for notch
    b, a = butter(order, [low, high], btype="bandstop")
    return filtfilt(b, a, data, axis=0)


def filter_forceplate(
    fp_df: pl.DataFrame,
    lowpass_hz: float = DEFAULT_FP_LOWPASS_HZ,
    lowpass_order: int = DEFAULT_FP_LOWPASS_ORDER,
    notch_center_hz: float = NOTCH_CENTER_HZ,
    notch_width_hz: float = NOTCH_WIDTH_HZ,
) -> pl.DataFrame:
    """Filter force plate data with notch + lowpass.

    Parameters
    ----------
    fp_df : pl.DataFrame
        Force plates DataFrame (long format) with columns:
        time, frame, fp_name, variable, axis, value.
    lowpass_hz : float
        Lowpass cutoff frequency.
    lowpass_order : int
        Lowpass filter order.
    notch_center_hz : float
        Notch center frequency.
    notch_width_hz : float
        Notch width.

    Returns
    -------
    pl.DataFrame
        Filtered force plates DataFrame.
    """
    # Calculate frame rate
    frames = fp_df["frame"].unique().sort()
    if len(frames) < 2:
        return fp_df

    time_vals = fp_df.filter(
        pl.col("frame") == frames[0]
    )["time"].to_list()

    if len(time_vals) < 2:
        return fp_df

    frame_rate = 1.0 / (time_vals[1] - time_vals[0])

    # Group by fp_name, variable, axis and filter each
    def _filter_group(group_df: pl.DataFrame) -> pl.DataFrame:
        values = group_df["value"].to_numpy()

        # Apply notch filter
        filtered = notch_filter(values, notch_center_hz, notch_width_hz, frame_rate)

        # Apply lowpass filter
        filtered = butterworth_filter(filtered, lowpass_hz, frame_rate, lowpass_order)

        return group_df.with_columns(pl.Series("value", filtered))

    result = fp_df.group_by("fp_name", "variable", "axis", maintain_order=True).map_groups(
        lambda g: _filter_group(g)
    )

    logger.info(f"Filtered force plates: {notch_center_hz}±{notch_width_hz/2} Hz notch + {lowpass_hz} Hz lowpass")
    return result


def filter_forceplate_wide(
    fp_df: pl.DataFrame,
    lowpass_hz: float = DEFAULT_FP_LOWPASS_HZ,
    lowpass_order: int = DEFAULT_FP_LOWPASS_ORDER,
    notch_center_hz: float = NOTCH_CENTER_HZ,
    notch_width_hz: float = NOTCH_WIDTH_HZ,
) -> pl.DataFrame:
    """Filter force plate data and return in wide format for OpenSim.

    Parameters
    ----------
    fp_df : pl.DataFrame
        Force plates DataFrame (long format).
    lowpass_hz : float
        Lowpass cutoff frequency.
    lowpass_order : int
        Lowpass filter order.
    notch_center_hz : float
        Notch center frequency.
    notch_width_hz : float
        Notch width.

    Returns
    -------
    pl.DataFrame
        Filtered force plates in wide format suitable for MOT writing.
    """
    filtered = filter_forceplate(fp_df, lowpass_hz, lowpass_order, notch_center_hz, notch_width_hz)

    # Pivot to wide format: one row per frame, columns for each fp_name/variable/axis
    wide = filtered.pivot(
        on=["fp_name", "variable", "axis"],
        index=["frame", "time"],
        values="value",
    )

    return wide
