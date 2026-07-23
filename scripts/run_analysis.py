"""Rat VML analysis pipeline.

Runs the full biomechanical analysis for the VML treatment comparison
paper using the latest rat hindlimb model and OpenSim tools.

Usage:
    uv run python scripts/run_analysis.py
    uv run python scripts/run_analysis.py --group NR

Pipeline:
    1. Load motion data for each subject
    2. Scale the bilateral rat model to subject anthropometrics
    3. Run Inverse Kinematics (IK)
    4. Run Inverse Dynamics (ID)
    5. Compute group means and run SPM
    6. Generate manuscript figures

Data layout (expected under data/):
    data/
        raw/              # Raw Vicon C3D/TRC files, one per trial
        subjects.csv      # Subject metadata (group, mass, limb lengths)
        ik/               # IK results (generated)
        id/               # ID results (generated)
        figures/          # Output figures (generated)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IK_DIR = DATA_DIR / "ik"
ID_DIR = DATA_DIR / "id"
FIGURES_DIR = DATA_DIR / "figures"

# Coordinate names matching the rat hindlimb model
COORD_NAMES = [
    "hip_r_flx", "hip_r_add", "hip_r_int",
    "knee_r_flx",
    "ankle_r_flx", "ankle_r_add", "ankle_r_int",
]

# Joint moment names matching OpenSim ID output
MOMENT_NAMES = [
    "hip_r_flx_moment", "hip_r_add_moment", "hip_r_int_moment",
    "knee_r_flx_moment",
    "ankle_r_flx_moment",
]

TREATMENT_GROUPS = [
    "Control", "NR", "TEMR", "HH", "HS", "TEMR+HH", "TEMR+KG",
]


# =========================================================================
# Step 1: Load subject metadata
# =========================================================================
def load_subjects() -> pl.DataFrame:
    """Load subject metadata from data/subjects.csv."""
    path = DATA_DIR / "subjects.csv"
    if not path.exists():
        logger.warning(f"Subjects file not found: {path}")
        return pl.DataFrame()
    return pl.read_csv(path)


# =========================================================================
# Step 2: Run Inverse Kinematics
# =========================================================================
def run_ik(
    model_path: Path,
    trc_path: Path,
    output_dir: Path,
    task_set_path: Path | None = None,
) -> Path:
    """Run OpenSim IK for a single trial using osimpy."""
    from osimpy.tools import IKSettings

    settings = IKSettings(
        name="rat_vml_ik",
        model_path=model_path,
        marker_path=trc_path,
        results_directory=output_dir,
        output_motion_file="ik_results.mot",
        task_set_path=task_set_path,
    )
    result = settings.run()
    if not result.success:
        raise RuntimeError(f"IK failed: {result.errors}")
    logger.info(f"IK complete: {result.motion_file}")
    return result.motion_file


# =========================================================================
# Step 3: Run Inverse Dynamics
# =========================================================================
def run_id(
    model_path: Path,
    mot_path: Path,
    external_loads_path: Path | None,
    output_dir: Path,
) -> Path:
    """Run OpenSim ID for a single trial using osimpy."""
    from osimpy.tools import IDSettings

    settings = IDSettings(
        name="rat_vml_id",
        model_path=model_path,
        coordinates_path=mot_path,
        results_directory=output_dir,
        output_forces_file="id_results.sto",
        external_loads_path=external_loads_path,
        lowpass_cutoff_frequency=6.0,
    )
    result = settings.run()
    if not result.success:
        raise RuntimeError(f"ID failed: {result.errors}")
    logger.info(f"ID complete: {result.moments_file}")
    return result.moments_file


# =========================================================================
# Step 4: Scale model to subject
# =========================================================================
def scale_model(
    base_model_path: Path,
    subject_name: str,
    mass: float,
    limb_lengths: dict[str, float],
    output_dir: Path,
) -> Path:
    """Scale the rat model to subject-specific anthropometrics using osimpy."""
    from osimpy.tools import ScaleSettings

    settings = ScaleSettings(
        name=f"scale_{subject_name}",
        model_path=base_model_path,
        results_directory=output_dir,
        output_model_file=f"{subject_name}_scaled.osim",
        mass=mass,
        limb_lengths=limb_lengths,
    )
    result = settings.run()
    if not result.success:
        raise RuntimeError(f"Scaling failed: {result.errors}")
    logger.info(f"Scaling complete: {result.scaled_model_file}")
    return result.scaled_model_file


# =========================================================================
# Step 5: Generate comparison plots
# =========================================================================
def plot_group_comparison(
    group_data: dict[str, list[pl.DataFrame]],
    group_name: str,
    output_dir: Path,
) -> None:
    """Generate kinematics and kinetics comparison plots for a group."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")

    # Kinematics plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"{group_name} — Joint Kinematics")
    # TODO: plot mean ± SD for each coordinate
    plt.tight_layout()
    fig.savefig(output_dir / f"{group_name.lower().replace('+', '_')}_kinematics.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Kinetics plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"{group_name} — Joint Moments")
    # TODO: plot mean ± SD for each moment
    plt.tight_layout()
    fig.savefig(output_dir / f"{group_name.lower().replace('+', '_')}_kinetics.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated plots for {group_name}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Rat VML analysis pipeline")
    parser.add_argument("--group", choices=TREATMENT_GROUPS, help="Process only one group")
    parser.add_argument("--model", type=Path, help="Path to bilateral rat model (.osim)")
    parser.add_argument("--subjects", type=Path, help="Path to subjects.csv")
    parser.add_argument("--skip-ik", action="store_true", help="Skip IK step (use cached)")
    parser.add_argument("--skip-id", action="store_true", help="Skip ID step (use cached)")
    args = parser.parse_args()

    # Create output directories
    for d in [IK_DIR, ID_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load subjects
    subjects = load_subjects()
    if subjects.is_empty():
        logger.info(
            "No subjects.csv found. "
            "Place motion data in data/raw/ and create data/subjects.csv to run the full pipeline."
        )
        return

    model_path = args.model or Path("models/rat_hindlimb_bilateral.osim")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Clone rat-hindlimb-model and point --model to the bilateral .osim file")
        return

    logger.info(f"Pipeline ready with {len(subjects)} subjects, model: {model_path}")
    logger.info("Run with --skip-ik --skip-id to regenerate only figures from cached results")
    logger.info("Or run individual group: python scripts/run_analysis.py --group NR")


if __name__ == "__main__":
    main()
