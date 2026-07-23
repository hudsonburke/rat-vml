"""Plotting functions for VML gait analysis.

Mirrors the MATLAB plotgroupspm.m and spmtimepointcomparison.m logic
from the UVA-MAMP-Lab Rats/Toolbox repos.

Key differences from default matplotlib plotting:
- Knee flexion angle is negated (MATLAB convention)
- Moments are normalized by mass*totalLength (Nm/kg)
- Stance/swing boundary at 50% is marked with a vertical line
- Plots show hip, knee, ankle in that order
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    import spm1d
except ImportError:
    spm1d = None


# Colour palette for treatment groups
GROUP_COLORS = {
    "Control": "#4C72B0",
    "No Repair": "#DD8452",
    "TEMR": "#55A868",
    "Healy Hydrogel": "#C44E52",
    "Healy Sponge": "#8172B3",
    "Healy Hydrogel + TEMR": "#937860",
    "Keratin Gel + TEMR": "#DA8BC3",
}


def _init_style():
    """Set up matplotlib style for manuscript-quality figures."""
    if plt is None:
        return
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        },
    )


def plot_kinematics(
    group_mean: np.ndarray,
    group_std: np.ndarray,
    control_mean: np.ndarray | None,
    control_std: np.ndarray | None,
    group_name: str,
    coord_names: list[str],
    output_path: Path,
    side: str = "r",
    n_points: int = 101,
    negated: list[str] | None = None,
) -> Path:
    """Generate kinematic comparison plot for one treatment group.

    Matches MATLAB plotgroupspm.m layout:
    - Hip flexion, Hip adduction, Hip internal rotation
    - Knee flexion (negated), Ankle dorsiflexion

    Parameters
    ----------
    group_mean, group_std : (202, n_coords) arrays
        Mean and std for the treatment group (stance+swing).
    control_mean, control_std : (202, n_coords) arrays or None
        Mean and std for the Control group.
    group_name : str
        Treatment group name (used in title and filename).
    coord_names : list[str]
        Column names matching the IK output columns.
    output_path : Path
        Directory to save the figure.
    side : str
        Side prefix ("r" or "l").
    n_points : int
        Points per phase (default 101).
    negated : list[str] or None
        Coordinates to negate (knee flexion by convention).
    """
    _init_style()
    if negated is None:
        negated = [f"knee_{side}_flx"]

    # Indices for the coordinates to plot
    plot_coords = [
        f"hip_{side}_flx",
        f"hip_{side}_add",
        f"hip_{side}_int",
        f"knee_{side}_flx",
        f"ankle_{side}_flx",
    ]
    titles = [
        "Hip Flexion",
        "Hip Adduction",
        "Hip Internal Rotation",
        "Knee Flexion",
        "Ankle Dorsiflexion",
    ]

    gait_pct = np.linspace(0, 200, 2 * n_points)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    fig.suptitle(f"{group_name} — Kinematics", fontsize=12)

    for i, (coord, title) in enumerate(zip(plot_coords, titles)):
        ax = axes[i]
        col_idx = coord_names.index(coord) if coord in coord_names else None
        if col_idx is None:
            ax.set_title(title)
            continue

        # Apply negation for knee flexion (MATLAB convention)
        sign = -1.0 if coord in negated else 1.0

        # Control reference
        if control_mean is not None and control_std is not None:
            c_mean = control_mean[:, col_idx] * sign
            c_std = control_std[:, col_idx]
            ax.plot(gait_pct, c_mean, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
            ax.fill_between(gait_pct, c_mean - c_std, c_mean + c_std,
                            color="gray", alpha=0.1)

        # Treatment group
        t_mean = group_mean[:, col_idx] * sign
        t_std = group_std[:, col_idx]
        color = GROUP_COLORS.get(group_name, "#4C72B0")
        ax.plot(gait_pct, t_mean, color=color, linewidth=2)
        ax.fill_between(gait_pct, t_mean - t_std, t_mean + t_std,
                        color=color, alpha=0.2)

        # Stance/swing boundary
        ax.axvline(x=n_points, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Gait %", fontsize=8)
        ax.set_ylabel("Angle (°)", fontsize=8)
        ax.set_xlim(0, 2 * n_points)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = Path(output_path) / f"{group_name.lower().replace('+', '_')}_kinematics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_kinetics(
    group_mean: np.ndarray,
    group_std: np.ndarray,
    control_mean: np.ndarray | None,
    control_std: np.ndarray | None,
    group_name: str,
    moment_names: list[str],
    output_path: Path,
    side: str = "r",
    n_points: int = 101,
    negated: list[str] | None = None,
) -> Path:
    """Generate joint-moment comparison plot for one treatment group."""
    _init_style()
    if negated is None:
        negated = [f"knee_{side}_flx_moment"]

    plot_moments = [
        f"hip_{side}_flx_moment",
        f"hip_{side}_add_moment",
        f"hip_{side}_int_moment",
        f"knee_{side}_flx_moment",
        f"ankle_{side}_flx_moment",
    ]
    titles = [
        "Hip Flexion Moment",
        "Hip Adduction Moment",
        "Hip Internal Rotation Moment",
        "Knee Flexion Moment",
        "Ankle Flexion Moment",
    ]

    gait_pct = np.linspace(0, 200, 2 * n_points)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    fig.suptitle(f"{group_name} — Joint Moments", fontsize=12)

    for i, (moment, title) in enumerate(zip(plot_moments, titles)):
        ax = axes[i]
        col_idx = moment_names.index(moment) if moment in moment_names else None
        if col_idx is None:
            ax.set_title(title)
            continue

        sign = -1.0 if moment in negated else 1.0

        # Control reference
        if control_mean is not None and control_std is not None:
            c_mean = control_mean[:, col_idx] * sign
            c_std = control_std[:, col_idx]
            ax.plot(gait_pct, c_mean, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
            ax.fill_between(gait_pct, c_mean - c_std, c_mean + c_std,
                            color="gray", alpha=0.1)

        # Treatment group
        t_mean = group_mean[:, col_idx] * sign
        t_std = group_std[:, col_idx]
        color = GROUP_COLORS.get(group_name, "#4C72B0")
        ax.plot(gait_pct, t_mean, color=color, linewidth=2)
        ax.fill_between(gait_pct, t_mean - t_std, t_mean + t_std,
                        color=color, alpha=0.2)

        ax.axvline(x=n_points, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Gait %", fontsize=8)
        ax.set_ylabel("Moment (Nm/kg)", fontsize=8)
        ax.set_xlim(0, 2 * n_points)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = Path(output_path) / f"{group_name.lower().replace('+', '_')}_kinetics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def generate_all_figures(
    group_results: dict[str, "GroupResult"],
    control_group: str,
    output_dir: Path,
    coord_names: list[str],
    moment_names: list[str],
) -> list[Path]:
    """Generate kinematics + kinetics figures for every treatment group.

    Returns list of saved figure paths.
    """
    from .pipeline import GroupResult

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    control = group_results.get(control_group)

    paths = []
    for name, grp in group_results.items():
        if name == control_group:
            continue
        if grp.ik_mean is None or grp.ik_std is None:
            logger.warning(f"Skipping {name}: no IK results")
            continue

        ref_mean = control.ik_mean if control else None
        ref_std = control.ik_std if control else None
        p = plot_kinematics(
            grp.ik_mean, grp.ik_std, ref_mean, ref_std,
            name, coord_names, output_dir,
        )
        paths.append(p)

        if grp.id_mean is not None and grp.id_std is not None:
            ref_mean = control.id_mean if control else None
            ref_std = control.id_std if control else None
            p = plot_kinetics(
                grp.id_mean, grp.id_std, ref_mean, ref_std,
                name, moment_names, output_dir,
            )
            paths.append(p)

    return paths
