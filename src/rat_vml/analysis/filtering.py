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
    time_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Filter marker positions with Butterworth lowpass.

    Works on long-format markers (frame, time, marker_name, x, y, z).
    Filters each marker's x, y, z jointly.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Long-format marker data.
    cutoff_hz : float
        Lowpass cutoff frequency in Hz.
    order : int
        Butterworth filter order.
    time_range : tuple[float, float] or None
        If provided, trim data to (start_time, end_time) before filtering.
    """
    # Trim to time range if specified
    if time_range is not None:
        start_time, end_time = time_range
        markers_df = markers_df.filter(
            (pl.col("time") >= start_time) & (pl.col("time") <= end_time)
        )
        if markers_df.is_empty():
            logger.warning(f"No data in time range {start_time}-{end_time}")
            return markers_df

    # Calculate frame rate from time spacing
    frames = markers_df["frame"].unique().sort()
    if len(frames) < 2:
        return markers_df

    time_vals = markers_df.filter(
        pl.col("frame") == frames[0]
    )["time"].unique().sort().to_list()

    if len(time_vals) < 2:
        return markers_df

    frame_rate = 1.0 / (time_vals[1] - time_vals[0])

    # Pivot to wide format for vectorized filtering
    # Columns will be: frame, time, x_MARKERA, y_MARKERA, z_MARKERA, x_MARKERB, ...
    wide = markers_df.pivot(
        on="marker_name",
        index=["frame", "time"],
        values=["x", "y", "z"],
    )

    # Get marker columns (excluding frame and time)
    marker_cols = [c for c in wide.columns if c not in ("frame", "time")]
    
    # Extract numpy array for vectorized filtering
    xyz_data = wide.select(marker_cols).to_numpy()
    valid_mask = ~np.isnan(xyz_data)
    
    # Apply filter to each column (vectorized across all markers)
    filtered_data = np.full_like(xyz_data, np.nan)
    for col_idx in range(xyz_data.shape[1]):
        channel = xyz_data[:, col_idx]
        col_valid = valid_mask[:, col_idx]
        
        if col_valid.sum() < 3:
            continue
        
        try:
            filt_channel = butterworth_filter(channel, cutoff_hz, frame_rate, order)
            filt_channel[~col_valid] = np.nan
            filtered_data[:, col_idx] = filt_channel
        except Exception:
            filtered_data[:, col_idx] = channel
    
    # Convert back to long format
    filtered_wide = wide.select(["frame", "time"]).hstack(
        pl.DataFrame(filtered_data, schema=marker_cols)
    )
    
    # Melt back to long format
    result = filtered_wide.melt(
        id_vars=["frame", "time"],
        variable_name="marker_col",
        value_name="value",
    )
    
    # Parse marker_col (e.g., 'x_RASI') into marker_name and axis
    result = result.with_columns([
        pl.col("marker_col").str.split("_").list.first().alias("axis"),
        pl.col("marker_col").str.split("_").list.slice(1).list.join("_").alias("marker_name"),
    ]).drop("marker_col")
    
    # Pivot axes to columns
    result = result.pivot(
        on="axis",
        index=["frame", "time", "marker_name"],
        values="value",
    )
    
    logger.info(f"Filtered markers at {cutoff_hz} Hz")
    return result


def markers_to_wide_trc(markers_df: pl.DataFrame) -> pl.DataFrame:
    """Convert long-format markers to wide-format TRC layout.

    Parameters
    ----------
    markers_df : pl.DataFrame
        Long-format marker data with columns: frame, time, marker_name, x, y, z.

    Returns
    -------
    pl.DataFrame
        Wide-format with columns: frame, time, {marker}_x, {marker}_y, {marker}_z
        sorted alphabetically by marker name.
    """
    # Pivot to wide format
    wide = markers_df.pivot(
        on="marker_name",
        index=["frame", "time"],
        values=["x", "y", "z"],
    )

    # Rename columns from x_MARKER to MARKER_x format (osimpy convention)
    rename_map = {}
    for col in wide.columns:
        if col.startswith("x_") or col.startswith("y_") or col.startswith("z_"):
            prefix, name = col.split("_", 1)
            rename_map[col] = f"{name}_{prefix}"
    wide = wide.rename(rename_map)

    # Sort columns alphabetically by marker name for TRC format
    marker_names = sorted(markers_df["marker_name"].unique().to_list())
    sorted_cols = ["frame", "time"]
    for name in marker_names:
        for suffix in ["_x", "_y", "_z"]:
            col = f"{name}{suffix}"
            if col in wide.columns:
                sorted_cols.append(col)
    return wide.select(sorted_cols)


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


def zero_outside_gait_cycle(
    fp_df: pl.DataFrame,
    events_df: pl.DataFrame,
) -> pl.DataFrame:
    """Zero force plate data outside the gait cycle window.

    If force plates have a 'side' column, zeros each plate outside its
    side's gait cycle. Otherwise, zeros all plates outside the overall window.
    """
    if events_df.is_empty():
        return fp_df

    # Get foot strike times for each side
    left_strikes = events_df.filter(
        (pl.col("context") == "Left") & (pl.col("label") == "Foot Strike")
    )["time"].to_list()

    right_strikes = events_df.filter(
        (pl.col("context") == "Right") & (pl.col("label") == "Foot Strike")
    )["time"].to_list()

    # Check if side column exists
    has_side = "side" in fp_df.columns

    if has_side:
        # Per-side zeroing
        def _zero_per_side(group_df: pl.DataFrame) -> pl.DataFrame:
            side = group_df["side"][0].lower() if group_df["side"][0] else "unknown"
            if side == "left" and left_strikes:
                start, end = min(left_strikes), max(left_strikes)
            elif side == "right" and right_strikes:
                start, end = min(right_strikes), max(right_strikes)
            else:
                # No side info or no strikes — use overall window
                all_strikes = left_strikes + right_strikes
                if not all_strikes:
                    return group_df
                start, end = min(all_strikes), max(all_strikes)

            return group_df.with_columns(
                pl.when((pl.col("time") < start) | (pl.col("time") > end))
                .then(0.0).otherwise(pl.col("value")).alias("value")
            )

        result = fp_df.group_by("fp_name", "trial_name", maintain_order=True).map_groups(
            lambda g: _zero_per_side(g)
        )
        logger.info("Zeroed force plates per side outside gait cycle")
    else:
        # Overall zeroing (no side info)
        all_strikes = left_strikes + right_strikes
        if not all_strikes:
            return fp_df

        gait_start = min(all_strikes)
        gait_end = max(all_strikes)

        result = fp_df.with_columns(
            pl.when((pl.col("time") < gait_start) | (pl.col("time") > gait_end))
            .then(0.0).otherwise(pl.col("value")).alias("value")
        )
        logger.info(f"Zeroed force plates outside gait cycle ({gait_start:.2f}-{gait_end:.2f}s)")

    return result
