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
