"""Marker filtering and TRC export for IK/ID.

Applies Butterworth lowpass filter to marker data and exports to TRC
format for OpenSim scaling and inverse kinematics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# Default filter parameters (matching MATLAB pipeline)
DEFAULT_CUTOFF_HZ = 15.0
DEFAULT_ORDER = 4


def butterworth_filter(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth lowpass filter (filtfilt).

    Parameters
    ----------
    data : np.ndarray
        Signal to filter. Can be 1D or 2D (samples, channels).
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate : float
        Sampling rate in Hz.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.
    """
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

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame (long format) with columns:
        frame, time, marker_name, x, y, z.
    cutoff_hz : float
        Cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    pl.DataFrame
        Filtered markers DataFrame.
    """
    # Get frame rate from time spacing
    frames = markers_df["frame"].unique().sort()
    if len(frames) < 2:
        return markers_df

    # Calculate frame rate from time column
    time_vals = markers_df.select(
        pl.col("time").filter(pl.col("frame") == frames[0])
    )["time"]
    if len(time_vals) < 2:
        return markers_df

    frame_rate = 1.0 / (time_vals[1] - time_vals[0])

    # Filter each marker separately
    marker_names = markers_df["marker_name"].unique().to_list()
    filtered_dfs = []

    for marker_name in marker_names:
        marker_data = markers_df.filter(pl.col("marker_name") == marker_name)

        # Extract x, y, z arrays
        x = marker_data["x"].to_numpy()
        y = marker_data["y"].to_numpy()
        z = marker_data["z"].to_numpy()

        # Stack for filtering
        xyz = np.column_stack([x, y, z])

        # Apply filter
        filtered_xyz = butterworth_filter(xyz, cutoff_hz, frame_rate, order)

        # Create filtered DataFrame
        filtered_marker = marker_data.with_columns([
            pl.Series("x", filtered_xyz[:, 0]),
            pl.Series("y", filtered_xyz[:, 1]),
            pl.Series("z", filtered_xyz[:, 2]),
        ])
        filtered_dfs.append(filtered_marker)

    result = pl.concat(filtered_dfs, how="diagonal")

    logger.info(f"Filtered {len(marker_names)} markers at {cutoff_hz} Hz")
    return result


def markers_to_wide_trc(markers_df: pl.DataFrame) -> pl.DataFrame:
    """Convert long-format markers to wide-format TRC layout.

    TRC format: one row per frame, columns for each marker's x, y, z.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Long-format markers with columns: frame, time, marker_name, x, y, z.

    Returns
    -------
    pl.DataFrame
        Wide-format DataFrame suitable for TRC writing.
    """
    # Pivot to wide format
    wide_df = markers_df.pivot(
        on="marker_name",
        index=["frame", "time"],
        values=["x", "y", "z"],
    )

    return wide_df


def prepare_trial_for_trc(
    markers_df: pl.DataFrame,
    events_df: pl.DataFrame | None = None,
    trial_name: str | None = None,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> pl.DataFrame | None:
    """Prepare a trial's marker data for TRC export.

    Filters markers and converts to wide format.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame (long format).
    events_df : pl.DataFrame or None
        Events DataFrame. If provided, filters to event window.
    trial_name : str or None
        Specific trial to prepare. If None, uses first trial.
    cutoff_hz : float
        Filter cutoff frequency.
    order : int
        Filter order.

    Returns
    -------
    pl.DataFrame or None
        Wide-format DataFrame ready for TRC writing, or None if no data.
    """
    # Filter to specific trial
    if trial_name:
        trial_markers = markers_df.filter(pl.col("trial_name") == trial_name)
    else:
        trial_name = markers_df["trial_name"].unique()[0]
        trial_markers = markers_df.filter(pl.col("trial_name") == trial_name)

    if trial_markers.is_empty():
        logger.warning(f"No marker data for trial {trial_name}")
        return None

    # Filter to event window if events provided
    if events_df is not None and not events_df.is_empty():
        trial_events = events_df.filter(pl.col("trial_name") == trial_name)
        if not trial_events.is_empty():
            first_time = trial_events["time"].min()
            last_time = trial_events["time"].max()
            trial_markers = trial_markers.filter(
                (pl.col("time") >= first_time) &
                (pl.col("time") <= last_time)
            )

    if trial_markers.is_empty():
        logger.warning(f"No marker data in event window for {trial_name}")
        return None

    # Apply filter
    filtered = filter_markers(trial_markers, cutoff_hz, order)

    # Convert to wide format
    wide = markers_to_wide_trc(filtered)

    return wide


def prepare_static_trial_for_trc(
    markers_df: pl.DataFrame,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> pl.DataFrame | None:
    """Prepare static trial marker data for TRC export.

    Filters markers, finds a clean frame, and returns wide format.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Static trial markers DataFrame.
    cutoff_hz : float
        Filter cutoff frequency.
    order : int
        Filter order.

    Returns
    -------
    pl.DataFrame or None
        Single-frame wide-format DataFrame for TRC writing.
    """
    from .static_trial import find_clean_frame, REQUIRED_MARKERS

    # Filter to static trial
    static_mask = pl.col("trial_name").str.to_lowercase().str.contains("static")
    static_df = markers_df.filter(static_mask)

    if static_df.is_empty():
        logger.warning("No static trial found")
        return None

    # Use last static trial
    trial_name = sorted(static_df["trial_name"].unique().to_list(), reverse=True)[0]
    static_df = static_df.filter(pl.col("trial_name") == trial_name)

    # Find clean frame
    clean_frame = find_clean_frame(static_df)
    if clean_frame is None:
        logger.warning("No clean frame in static trial")
        return None

    # Get single frame
    frame_df = static_df.filter(pl.col("frame") == clean_frame)

    # Filter (single frame, so filter does nothing but keeps interface consistent)
    filtered = filter_markers(frame_df, cutoff_hz, order)

    # Convert to wide format
    wide = markers_to_wide_trc(filtered)

    return wide
