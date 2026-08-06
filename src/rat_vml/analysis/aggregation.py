"""Aggregation and comparison functions for rat-vml analysis.

Provides parameterized functions for:
- Aggregating IK/ID/Moco results by group and session/timepoint
- Comparing groups at a given timepoint
- Comparing timepoints within a group
- SPM t-tests for statistical comparison

Usage::

    from rat_vml.analysis.aggregation import (
        load_results,
        aggregate_by_group,
        compare_groups,
        compare_timepoints,
        spm_ttest,
    )

    # Load IK results
    results = load_results("data/processed", result_type="ik")

    # Aggregate Control vs TEMR at week24
    control = aggregate_by_group(results, group="control", session="week24")
    temr = aggregate_by_group(results, group="temr", session="week24")

    # Compare
    diff = compare_groups(control, temr)

    # SPM t-test
    t = spm_ttest(control, temr, coord="knee_angle_r")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    """Aggregated results for one group at one session/timepoint."""

    group: str
    session: str
    coord_names: list[str]

    # Per-subject means: dict[subject_id, array(n_coords, n_points)]
    subject_means: dict[str, np.ndarray] = field(default_factory=dict)

    # Group-level stats (computed by aggregate_by_group)
    mean: np.ndarray | None = None  # (n_coords, n_points)
    std: np.ndarray | None = None   # (n_coords, n_points)
    n_subjects: int = 0

    # Metadata
    subjects: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing two groups or timepoints."""

    group_a: str
    group_b: str
    session: str
    coord_names: list[str]

    # Group means and stds
    mean_a: np.ndarray  # (n_coords, n_points)
    std_a: np.ndarray
    mean_b: np.ndarray
    std_b: np.ndarray

    # SPM results per coordinate
    spm_results: dict[str, object] = field(default_factory=dict)


def load_results(
    data_dir: str | Path,
    result_type: str = "ik",
    session: str | None = None,
    subject_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Load results from Parquet files using glob.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing subject subdirectories with Parquet files.
    result_type : str
        Type of results to load: "ik", "id", or "moco".
    session : str, optional
        Filter to a specific session.
    subject_ids : list[str], optional
        Filter to specific subjects.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
        time, coord, value, subject_id, session_id, trial_name
    """
    import glob as globmod

    data_dir = Path(data_dir)
    pattern = str(data_dir / "*" / f"{result_type}_results.parquet")
    paths = sorted(globmod.glob(pattern))

    if not paths:
        return pl.DataFrame()

    dfs = []
    for path in paths:
        p = Path(path)
        subject_id = p.parent.name

        if subject_ids and subject_id not in subject_ids:
            continue

        df = pl.read_parquet(path)

        if session:
            df = df.filter(pl.col("session_id") == session)

        if not df.is_empty():
            dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs)


def aggregate_by_group(
    results: pl.DataFrame,
    group: str,
    session: str,
    n_points: int = 101,
) -> GroupResult:
    """Aggregate results by group and session.

    Computes per-subject means across trials, then group mean ± std.

    Parameters
    ----------
    results : pl.DataFrame
        Long-format results from load_results().
    group : str
        Treatment group name.
    session : str
        Session/timepoint name.
    n_points : int
        Number of points per curve (for splining).

    Returns
    -------
    GroupResult
        Aggregated results with per-subject means and group stats.
    """
    from .subject_groups import get_group

    # Get unique coordinates
    coord_names = sorted(results["coord"].unique().to_list())

    # Filter to this group/session
    filtered = results.filter(pl.col("session_id") == session)

    # Get subjects in this group
    subjects_in_group = [
        sid for sid in filtered["subject_id"].unique().to_list()
        if get_group(sid) == group
    ]

    if not subjects_in_group:
        logger.warning(f"No subjects found for group '{group}' at session '{session}'")
        return GroupResult(group=group, session=session, coord_names=coord_names)

    # Compute per-subject means across trials
    subject_means = {}
    for subject_id in subjects_in_group:
        subject_data = filtered.filter(pl.col("subject_id") == subject_id)

        # Pivot to wide format: time × coord
        subject_arrays = []
        for coord in coord_names:
            coord_data = subject_data.filter(pl.col("coord") == coord)

            if coord_data.is_empty():
                subject_arrays.append(np.zeros(n_points))
                continue

            # Average across trials for this coordinate
            trial_means = []
            for trial in coord_data["trial_name"].unique().to_list():
                trial_data = coord_data.filter(pl.col("trial_name") == trial)
                values = trial_data.sort("time")["value"].to_numpy()

                # Spline to n_points if needed
                if len(values) != n_points:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(values))
                    x_new = np.linspace(0, 1, n_points)
                    f = interp1d(x_old, values, kind="linear")
                    values = f(x_new)

                trial_means.append(values)

            subject_arrays.append(np.mean(trial_means, axis=0))

        subject_means[subject_id] = np.array(subject_arrays)

    # Compute group stats
    all_subjects = np.stack(list(subject_means.values()))
    group_mean = np.mean(all_subjects, axis=0)
    group_std = np.std(all_subjects, axis=0, ddof=1) if len(subjects_in_group) > 1 else np.zeros_like(group_mean)

    return GroupResult(
        group=group,
        session=session,
        coord_names=coord_names,
        subject_means=subject_means,
        mean=group_mean,
        std=group_std,
        n_subjects=len(subjects_in_group),
        subjects=subjects_in_group,
    )


def compare_groups(
    group_a: GroupResult,
    group_b: GroupResult,
) -> ComparisonResult:
    """Compare two groups at the same session/timepoint.

    Parameters
    ----------
    group_a, group_b : GroupResult
        Aggregated results from aggregate_by_group().

    Returns
    -------
    ComparisonResult
        Comparison with SPM t-test results per coordinate.
    """
    if group_a.session != group_b.session:
        logger.warning(f"Sessions differ: {group_a.session} vs {group_b.session}")

    # SPM t-tests per coordinate
    spm_results = {}
    for i, coord in enumerate(group_a.coord_names):
        # Get per-subject curves for this coordinate
        a_curves = np.stack([
            group_a.subject_means[sid][i]
            for sid in group_a.subjects
        ])
        b_curves = np.stack([
            group_b.subject_means[sid][i]
            for sid in group_b.subjects
        ])

        try:
            import spm1d
            t = spm1d.stats.ttest2(a_curves, b_curves)
            ti = t.inference(alpha=0.05)
            spm_results[coord] = ti
        except ImportError:
            logger.warning("spm1d not installed, skipping SPM t-test")
            spm_results[coord] = None

    return ComparisonResult(
        group_a=group_a.group,
        group_b=group_b.group,
        session=group_a.session,
        coord_names=group_a.coord_names,
        mean_a=group_a.mean,
        std_a=group_a.std,
        mean_b=group_b.mean,
        std_b=group_b.std,
        spm_results=spm_results,
    )


def compare_timepoints(
    group: GroupResult,
    session_a: str,
    session_b: str,
) -> ComparisonResult:
    """Compare two timepoints within the same group.

    Parameters
    ----------
    group : GroupResult
        Aggregated results for one group.
    session_a, session_b : str
        Session/timepoint names to compare.

    Returns
    -------
    ComparisonResult
        Comparison with SPM paired t-test results.
    """
    # This requires loading results for both sessions
    # For now, return a placeholder
    raise NotImplementedError(
        "compare_timepoints requires loading results for both sessions. "
        "Use load_results() with session filter, then aggregate_by_group() for each."
    )


def spm_ttest(
    group_a: GroupResult,
    group_b: GroupResult,
    coord: str,
    alpha: float = 0.05,
) -> object:
    """Run SPM t-test for one coordinate between two groups.

    Parameters
    ----------
    group_a, group_b : GroupResult
        Aggregated results.
    coord : str
        Coordinate name (e.g., "knee_angle_r").
    alpha : float
        Significance level.

    Returns
    -------
    spm1d.stats.ttest2 result
        SPM inference result.
    """
    try:
        import spm1d
    except ImportError:
        raise ImportError("spm1d is required for SPM t-tests: pip install spm1d")

    i = group_a.coord_names.index(coord)

    a_curves = np.stack([group_a.subject_means[sid][i] for sid in group_a.subjects])
    b_curves = np.stack([group_b.subject_means[sid][i] for sid in group_b.subjects])

    t = spm1d.stats.ttest2(a_curves, b_curves)
    return t.inference(alpha=alpha)
