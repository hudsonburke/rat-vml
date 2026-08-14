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
) -> Path:
    """Plot muscle activation profiles (0 to 1) for individual muscles.

    Parameters
    ----------
    activation_data : np.ndarray
        Shape (n_timepoints, n_muscles) with values in [0, 1].
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
