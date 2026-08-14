"""SPM 1D analysis for gait data.

Implements Statistical Parametric Mapping (SPM) for comparing
biomechanical data across groups and timepoints.

Mirrors the MATLAB spmtimepointcomparison.m and plotSPM.m logic
from the UVA-MAMP-Lab/Toolbox repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

try:
    import spm1d
    _HAS_SPM1D = True
except ImportError:
    _HAS_SPM1D = False
    logger.warning("spm1d not installed — SPM analysis unavailable")


@dataclass
class SPMResult:
    """Result of an SPM 1D analysis."""
    significant: bool
    clusters: list[tuple[int, int]]  # List of (start, end) indices
    t_values: np.ndarray
    p_values: np.ndarray
    alpha: float = 0.05


def spm_ttest_1d(
    data1: np.ndarray,
    data2: np.ndarray,
    alpha: float = 0.05,
    paired: bool = True,
) -> SPMResult:
    """Run SPM 1D t-test between two groups.

    Parameters
    ----------
    data1 : np.ndarray
        Shape (n_subjects, n_timepoints) for group 1.
    data2 : np.ndarray
        Shape (n_subjects, n_timepoints) for group 2.
    alpha : float
        Significance level (default 0.05).
    paired : bool
        If True, run paired t-test; otherwise unpaired.

    Returns
    -------
    SPMResult
        Significant clusters and t/p values.
    """
    if not _HAS_SPM1D:
        return SPMResult(significant=False, clusters=[], t_values=np.array([]), p_values=np.array([]))

    # Remove NaN rows (incomplete pairs)
    valid_mask = ~(np.isnan(data1).any(axis=1) | np.isnan(data2).any(axis=1))
    d1 = data1[valid_mask]
    d2 = data2[valid_mask]

    if len(d1) < 3:
        logger.warning("Too few complete pairs for SPM analysis")
        return SPMResult(significant=False, clusters=[], t_values=np.array([]), p_values=np.array([]))

    # Run SPM
    if paired:
        t = spm1d.stats.ttest_paired(d1, d2)
    else:
        t = spm1d.stats.ttest2(d1, d2)

    inference = t.inference(alpha, two_tailed=True, interp=True)

    clusters = []
    for cluster in inference.clusters:
        clusters.append((cluster.endpoints[0], cluster.endpoints[1]))

    return SPMResult(
        significant=inference.nClusters > 0,
        clusters=clusters,
        t_values=inference.t,
        p_values=inference.p,
        alpha=alpha,
    )


def spm_timepoint_comparison(
    data: np.ndarray,
    variable_names: list[str],
    labels: list[str],
    timepoints: list[str] | None = None,
    negated: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, SPMResult]:
    """Compare baseline vs other timepoints using SPM.

    Mirrors MATLAB spmtimepointcomparison.m.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_timepoints, n_variables, n_trials, n_sessions).
    variable_names : list[str]
        Names of all variables in the data.
    labels : list[str]
        Which variables to plot/analyze.
    timepoints : list[str] or None
        Timepoint names (e.g., ["Baseline", "Week12"]).
    negated : list[str] or None
        Variables to negate (e.g., knee flexion).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Mapping of label -> SPMResult for each variable.
    """
    if timepoints is None:
        timepoints = [f"T{i}" for i in range(data.shape[3])]

    # Negate specified variables
    if negated:
        for neg_name in negated:
            if neg_name in variable_names:
                idx = variable_names.index(neg_name)
                data[:, idx, :, :] = -data[:, idx, :, :]

    results = {}
    for label in labels:
        if label not in variable_names:
            continue
        idx = variable_names.index(label)

        # Baseline vs each other timepoint
        baseline = data[:, idx, :, 0]  # (n_timepoints, n_trials)
        for t_idx in range(1, data.shape[3]):
            comparison = data[:, idx, :, t_idx]
            result = spm_ttest_1d(baseline.T, comparison.T, alpha=alpha)
            results[f"{label}_vs_{timepoints[t_idx]}"] = result

    return results


def find_gait_cycles(
    events_df,
    subject_id: str,
    session: str,
    trial: str,
) -> tuple[float, float]:
    """Find first foot strike and last foot strike for a trial.

    Returns (start_time, end_time) of the gait cycle window.
    """
    trial_events = events_df.filter(
        (pl.col("subject_id") == subject_id) &
        (pl.col("session_id") == session) &
        (pl.col("trial_name") == trial)
    )

    strikes = trial_events.filter(pl.col("label") == "Foot Strike")
    if strikes.is_empty():
        return 0.0, 0.0

    return float(strikes["time"].min()), float(strikes["time"].max())


# ---------------------------------------------------------------------------
# Moco activation plotting
# ---------------------------------------------------------------------------

def plot_activation(
    activation_data: np.ndarray,
    muscle_names: list[str],
    time: np.ndarray,
    output_path: Path,
    title: str = "Muscle Activations",
    group_name: str | None = None,
    spm_result: SPMResult | None = None,
    stance_swing: bool = True,
    n_points: int = 101,
) -> Path:
    """Plot muscle activation profiles (0 to 1) for individual muscles.

    Parameters
    ----------
    activation_data : np.ndarray
        Shape (n_timepoints, n_muscles) with values in [0, 1].
        If stance_swing=True, first n_points are stance, rest are swing.
    muscle_names : list[str]
        Names for each muscle column.
    time : np.ndarray
        Time points (gait %, 0-200 for stance+swing).
    output_path : Path
        Directory to save figure.
    title : str
        Figure title.
    group_name : str or None
        Group name for filename.
    spm_result : SPMResult or None
        SPM result for highlighting significant regions.
    stance_swing : bool
        If True, show stance/swing boundary and label x-axis accordingly.
    n_points : int
        Points per phase (default 101).

    Returns
    -------
    Path
        Path to saved figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib not installed")
        return Path()

    sns.set_theme(style="ticks", context="paper", rc={
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "font.family": "sans-serif",
    })

    n_muscles = len(muscle_names)
    n_cols = min(4, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=12)

    for idx, muscle in enumerate(muscle_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Plot activation
        ax.plot(time, activation_data[:, idx], linewidth=1.5, color="#4C72B0")
        ax.fill_between(time, 0, activation_data[:, idx], alpha=0.2, color="#4C72B0")

        # SPM highlighting
        if spm_result is not None and spm_result.significant:
            for start, end in spm_result.clusters:
                ax.axvspan(start, end, color="red", alpha=0.2, zorder=0)

        ax.set_title(muscle, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(labelsize=7)

        if row == n_rows - 1:
            if stance_swing:
                ax.set_xlabel("Stance % | Swing %", fontsize=8)
            else:
                ax.set_xlabel("Gait %", fontsize=8)
        if col == 0:
            ax.set_ylabel("Activation", fontsize=8)

    # Hide unused axes
    for idx in range(n_muscles, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if group_name:
        fname = f"{group_name.lower().replace('+', '_')}_activations.png"
    else:
        fname = "activations.png"
    path = output_path / fname
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_activation_comparison(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    muscle_names: list[str],
    time: np.ndarray,
    output_path: Path,
    group_name: str,
    spm_results: dict[str, SPMResult] | None = None,
    stance_swing: bool = True,
    n_points: int = 101,
) -> Path:
    """Plot activation comparison between control and treatment groups.

    Parameters
    ----------
    control_data : np.ndarray
        Shape (n_timepoints, n_muscles) for control group.
    treatment_data : np.ndarray
        Shape (n_timepoints, n_muscles) for treatment group.
    muscle_names : list[str]
        Names for each muscle column.
    time : np.ndarray
        Time points (gait %).
    output_path : Path
        Directory to save figure.
    group_name : str
        Treatment group name.
    spm_results : dict or None
        Mapping of muscle name -> SPMResult for highlighting.

    Returns
    -------
    Path
        Path to saved figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib not installed")
        return Path()

    sns.set_theme(style="ticks", context="paper", rc={
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "font.family": "sans-serif",
    })

    n_muscles = len(muscle_names)
    n_cols = min(4, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f"{group_name} — Muscle Activations", fontsize=12)

    for idx, muscle in enumerate(muscle_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Control (gray dashed)
        ax.plot(time, control_data[:, idx], linewidth=1.5, color="gray",
                linestyle="--", alpha=0.6)

        # Treatment (colored)
        color = GROUP_COLORS.get(group_name, "#4C72B0")
        ax.plot(time, treatment_data[:, idx], linewidth=1.5, color=color)
        ax.fill_between(time, 0, treatment_data[:, idx], alpha=0.15, color=color)

        # SPM highlighting
        spm_res = spm_results.get(muscle) if spm_results else None
        if spm_res is not None and spm_res.significant:
            for start, end in spm_res.clusters:
                ax.axvspan(start, end, color="red", alpha=0.2, zorder=0)

        ax.set_title(muscle, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(labelsize=7)

        if row == n_rows - 1:
            if stance_swing:
                ax.set_xlabel("Stance % | Swing %", fontsize=8)
            else:
                ax.set_xlabel("Gait %", fontsize=8)
        if col == 0:
            ax.set_ylabel("Activation", fontsize=8)

    for idx in range(n_muscles, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    path = output_path / f"{group_name.lower().replace('+', '_')}_activations.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Stance/swing splining
# ---------------------------------------------------------------------------

def spline_to_stance_swing(
    data: np.ndarray,
    times: np.ndarray,
    event_times: np.ndarray,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Spline data to separate stance and swing phases.

    Each phase is splined to n_points (default 101).
    """
    from scipy.interpolate import CubicSpline

    if len(event_times) < 3:
        return data, data

    fs1, fo1, fs2 = event_times[0], event_times[1], event_times[2]
    stance_pct = np.linspace(0, 100, n_points)
    swing_pct = np.linspace(0, 100, n_points)

    stance_mask = (times >= fs1) and (times <= fo1)
    swing_mask = (times >= fo1) and (times <= fs2)

    stance_times = times[stance_mask]
    swing_times = times[swing_mask]

    if len(stance_times) < 2 or len(swing_times) < 2:
        return data, data

    stance_norm = (stance_times - fs1) / (fo1 - fs1) * 100
    swing_norm = (swing_times - fo1) / (fs2 - fo1) * 100

    if data.ndim == 1:
        cs_stance = CubicSpline(stance_norm, data[stance_mask])
        cs_swing = CubicSpline(swing_norm, data[swing_mask])
        return cs_stance(stance_pct), cs_swing(swing_pct)
    else:
        stance_out = np.zeros((n_points, data.shape[1]))
        swing_out = np.zeros((n_points, data.shape[1]))
        for col in range(data.shape[1]):
            cs_stance = CubicSpline(stance_norm, data[stance_mask, col])
            cs_swing = CubicSpline(swing_norm, data[swing_mask, col])
            stance_out[:, col] = cs_stance(stance_pct)
            swing_out[:, col] = cs_swing(swing_pct)
        return stance_out, swing_out


def prepare_stance_swing_data(
    data: np.ndarray,
    times: np.ndarray,
    events_df,
    subject_id: str,
    session: str,
    trial: str,
    n_points: int = 101,
) -> np.ndarray:
    """Prepare data splined to stance+swing phases for a single trial.

    Returns array of shape (2*n_points, n_cols) with stance then swing.
    """
    trial_events = events_df.filter(
        (pl.col("subject_id") == subject_id)
        and (pl.col("session_id") == session)
        and (pl.col("trial_name") == trial)
    )

    strikes = trial_events.filter(pl.col("label") == "Foot Strike")
    offs = trial_events.filter(pl.col("label") == "Foot Off")

    if len(strikes) < 2 or len(offs) < 1:
        return data

    fs1 = float(strikes["time"].min())
    fo1 = float(offs["time"].min())
    fs2_series = strikes.filter(pl.col("time") > fo1)["time"]
    fs2 = float(fs2_series.min()) if len(fs2_series) > 0 else float(strikes["time"].max())

    event_times = np.array([fs1, fo1, fs2])
    stance, swing = spline_to_stance_swing(data, times, event_times, n_points)
    return np.vstack([stance, swing])


# ---------------------------------------------------------------------------
# Results aggregation for plotting
# ---------------------------------------------------------------------------

@dataclass
class GroupResult:
    """Aggregated results for a treatment group."""
    group_name: str
    ik_mean: np.ndarray | None = None   # (202, n_coords)
    ik_std: np.ndarray | None = None
    id_mean: np.ndarray | None = None   # (202, n_moments)
    id_std: np.ndarray | None = None
    activation_mean: np.ndarray | None = None  # (202, n_muscles)
    activation_std: np.ndarray | None = None
    n_subjects: int = 0
    n_trials: int = 0


def load_ik_results(
    data_dir: Path,
    subject_id: str,
    session: str,
    trial: str,
) -> np.ndarray | None:
    """Load IK results from Parquet file.

    Returns array of shape (n_frames, n_coords) or None.
    """
    import polars as pl

    path = data_dir / subject_id / f"ik_{session}_{trial}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)
    # Assuming columns: time, coord1, coord2, ...
    numeric_cols = [c for c in df.columns if c not in ("time", "frame")]
    return df.select(numeric_cols).to_numpy()


def load_id_results(
    data_dir: Path,
    subject_id: str,
    session: str,
    trial: str,
) -> np.ndarray | None:
    """Load ID results from Parquet file.

    Returns array of shape (n_frames, n_moments) or None.
    """
    import polars as pl

    path = data_dir / subject_id / f"id_{session}_{trial}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)
    numeric_cols = [c for c in df.columns if c not in ("time", "frame")]
    return df.select(numeric_cols).to_numpy()


def load_activation_results(
    data_dir: Path,
    subject_id: str,
    session: str,
    trial: str,
) -> np.ndarray | None:
    """Load Moco activation results from Parquet file.

    Returns array of shape (n_frames, n_muscles) or None.
    """
    import polars as pl

    path = data_dir / subject_id / f"moco_{session}_{trial}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)
    numeric_cols = [c for c in df.columns if c not in ("time", "frame")]
    return df.select(numeric_cols).to_numpy()


def aggregate_subject(
    data_dir: Path,
    subject_id: str,
    session: str,
    trials: list[str],
    events_df,
    result_type: str = "ik",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate results across trials for a single subject.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    subject_id : str
        Subject identifier.
    session : str
        Session name.
    trials : list[str]
        Trial names to include.
    events_df : pl.DataFrame
        Events data for spline timing.
    result_type : str
        "ik", "id", or "activation".

    Returns
    -------
    mean : np.ndarray
        Shape (202, n_vars).
    std : np.ndarray
        Shape (202, n_vars).
    """
    import polars as pl

    load_fn = {
        "ik": load_ik_results,
        "id": load_id_results,
        "activation": load_activation_results,
    }[result_type]

    all_splined = []
    for trial in trials:
        data = load_fn(data_dir, subject_id, session, trial)
        if data is None:
            continue

        # Get time array
        trial_path = data_dir / subject_id / f"{result_type}_{session}_{trial}.parquet"
        df = pl.read_parquet(trial_path)
        times = df["time"].to_numpy()

        # Spline to stance+swing
        splined = prepare_stance_swing_data(
            data, times, events_df, subject_id, session, trial
        )
        all_splined.append(splined)

    if not all_splined:
        n_vars = data.shape[1] if data is not None else 0
        return np.zeros((202, n_vars)), np.zeros((202, n_vars))

    stacked = np.stack(all_splined)  # (n_trials, 202, n_vars)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def aggregate_group(
    data_dir: Path,
    subject_ids: list[str],
    session: str,
    trials: list[str],
    events_df,
    group_name: str,
    result_type: str = "ik",
) -> GroupResult:
    """Aggregate results across subjects for a treatment group.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    subject_ids : list[str]
        Subjects in this group.
    session : str
        Session name.
    trials : list[str]
        Trial names to include.
    events_df : pl.DataFrame
        Events data.
    group_name : str
        Group name for labeling.
    result_type : str
        "ik", "id", or "activation".

    Returns
    -------
    GroupResult
        Aggregated mean/std across subjects.
    """
    all_subject_means = []
    total_trials = 0

    for subject_id in subject_ids:
        mean, std = aggregate_subject(
            data_dir, subject_id, session, trials, events_df, result_type
        )
        if mean.size > 0:
            all_subject_means.append(mean)
            total_trials += len(trials)

    if not all_subject_means:
        return GroupResult(group_name=group_name, n_subjects=0, n_trials=0)

    stacked = np.stack(all_subject_means)  # (n_subjects, 202, n_vars)
    group_mean = np.mean(stacked, axis=0)
    group_std = np.std(stacked, axis=0)

    result = GroupResult(
        group_name=group_name,
        n_subjects=len(all_subject_means),
        n_trials=total_trials,
    )

    if result_type == "ik":
        result.ik_mean = group_mean
        result.ik_std = group_std
    elif result_type == "id":
        result.id_mean = group_mean
        result.id_std = group_std
    elif result_type == "activation":
        result.activation_mean = group_mean
        result.activation_std = group_std

    return result


def aggregate_all_groups(
    data_dir: Path,
    group_subjects: dict[str, list[str]],
    session: str,
    trials: list[str],
    events_df,
    result_types: list[str] | None = None,
) -> dict[str, GroupResult]:
    """Aggregate results for all treatment groups.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    group_subjects : dict
        Mapping of group name -> list of subject IDs.
    session : str
        Session name.
    trials : list[str]
        Trial names to include.
    events_df : pl.DataFrame
        Events data.
    result_types : list[str] or None
        Which result types to aggregate (default: ["ik", "id"]).

    Returns
    -------
    dict
        Mapping of group name -> GroupResult.
    """
    if result_types is None:
        result_types = ["ik", "id"]

    results = {}
    for group_name, subject_ids in group_subjects.items():
        result = GroupResult(group_name=group_name, n_subjects=len(subject_ids))

        for rt in result_types:
            agg = aggregate_group(
                data_dir, subject_ids, session, trials,
                events_df, group_name, result_type=rt
            )
            if rt == "ik":
                result.ik_mean = agg.ik_mean
                result.ik_std = agg.ik_std
            elif rt == "id":
                result.id_mean = agg.id_mean
                result.id_std = agg.id_std
            elif rt == "activation":
                result.activation_mean = agg.activation_mean
                result.activation_std = agg.activation_std

            result.n_trials = max(result.n_trials, agg.n_trials)

        results[group_name] = result

    return results
