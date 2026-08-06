"""Plotting helpers for the Quarto manuscript.

These functions are imported in the .qmd file to generate figures
from the analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from rat_vml.analysis.aggregation import load_results, aggregate_by_group, compare_groups
from rat_vml.analysis.plots import GROUP_COLORS


def plot_group_kinematics(
    data_dir: str,
    group: str,
    session: str = "week24",
    control_group: str = "control",
    output_path: str | None = None,
    coord_names: list[str] | None = None,
) -> plt.Figure:
    """Plot kinematics for one group vs control.

    Parameters
    ----------
    data_dir : str
        Path to processed data directory.
    group : str
        Treatment group name.
    session : str
        Session/timepoint (default: week24).
    control_group : str
        Control group name (default: control).
    output_path : str, optional
        Save figure to this path.
    coord_names : list[str], optional
        Coordinates to plot (default: hip, knee, ankle angles).

    Returns
    -------
    plt.Figure
    """
    results = load_results(data_dir, result_type="ik", session=session)

    group_result = aggregate_by_group(results, group=group, session=session)
    control_result = aggregate_by_group(results, group=control_group, session=session)

    if group_result.n_subjects == 0:
        raise ValueError(f"No subjects found for group '{group}' at session '{session}'")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    if coord_names is None:
        coord_names = group_result.coord_names[:6]

    for i, coord in enumerate(coord_names):
        if i >= len(axes):
            break

        ax = axes[i]
        ci = group_result.coord_names.index(coord)

        # Control
        if control_result.n_subjects > 0:
            ax.fill_between(
                np.linspace(0, 100, 101),
                control_result.mean[ci] - control_result.std[ci],
                control_result.mean[ci] + control_result.std[ci],
                alpha=0.2, color=GROUP_COLORS.get(control_group, "gray"),
            )
            ax.plot(np.linspace(0, 100, 101), control_result.mean[ci],
                    color=GROUP_COLORS.get(control_group, "gray"),
                    label=control_group, linewidth=2)

        # Treatment group
        ax.fill_between(
            np.linspace(0, 100, 101),
            group_result.mean[ci] - group_result.std[ci],
            group_result.mean[ci] + group_result.std[ci],
            alpha=0.2, color=GROUP_COLORS.get(group, "blue"),
        )
        ax.plot(np.linspace(0, 100, 101), group_result.mean[ci],
                color=GROUP_COLORS.get(group, "blue"),
                label=group, linewidth=2)

        ax.set_title(coord.replace("_r", "").replace("_l", "").replace("_", " ").title())
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)

    plt.suptitle(f"{group} vs {control_group} Kinematics ({session})")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_group_comparison(
    data_dir: str,
    group_a: str,
    group_b: str,
    session: str = "week24",
    result_type: str = "ik",
    output_path: str | None = None,
) -> plt.Figure:
    """Plot SPM comparison between two groups.

    Parameters
    ----------
    data_dir : str
        Path to processed data directory.
    group_a, group_b : str
        Group names to compare.
    session : str
        Session/timepoint.
    result_type : str
        "ik" for kinematics, "id" for kinetics, "moco" for muscle analysis.
    output_path : str, optional
        Save figure to this path.

    Returns
    -------
    plt.Figure
    """
    results = load_results(data_dir, result_type=result_type, session=session)

    a = aggregate_by_group(results, group=group_a, session=session)
    b = aggregate_by_group(results, group=group_b, session=session)

    comparison = compare_groups(a, b)

    n_coords = len(comparison.coord_names)
    fig, axes = plt.subplots(n_coords, 1, figsize=(10, 3 * n_coords))

    if n_coords == 1:
        axes = [axes]

    for i, coord in enumerate(comparison.coord_names):
        ax = axes[i]

        # Plot group means
        x = np.linspace(0, 100, 101)
        ax.plot(x, comparison.mean_a[i], label=group_a, linewidth=2)
        ax.plot(x, comparison.mean_b[i], label=group_b, linewidth=2)

        # Plot SPM significant regions
        spm_result = comparison.spm_results.get(coord)
        if spm_result is not None and hasattr(spm_result, "clusters"):
            for cluster in spm_result.clusters:
                ax.axvspan(cluster.start, cluster.end, alpha=0.3, color="red")

        ax.set_title(coord.replace("_r", "").replace("_l", "").replace("_", " ").title())
        ax.set_xlabel("% Gait Cycle")
        ax.legend(fontsize=8)

    plt.suptitle(f"{group_a} vs {group_b} {result_type.upper()} ({session})")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
