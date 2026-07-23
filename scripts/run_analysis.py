"""Rat VML analysis pipeline.

Runs the full biomechanical analysis for the VML treatment comparison
paper using the latest rat hindlimb model and OpenSim tools.

Usage:
    uv run python scripts/run_analysis.py
    uv run python scripts/run_analysis.py --group NR --data-dir ../data/vml

Pipeline:
    1. Load subject metadata (group, mass, segment lengths)
    2. Scale the bilateral rat model to each subject's anthropometrics
       (rat_vml.analysis.pipeline.scale_subject — manual scale factors
        + Hicks regression for inertial properties)
    3. Run Inverse Kinematics (rat_vml.analysis.pipeline.run_ik)
    4. Run Inverse Dynamics (rat_vml.analysis.pipeline.run_id)
    5. Aggregate group results and run SPM
    6. Generate manuscript figures (rat_vml.analysis.plots)

The pipeline uses :mod:`rat_vml.analysis` which composes osimpy's
generic tool wrappers into rat-specific workflows.

Data layout:
    <data-dir>/
        raw/              # Raw Vicon C3D/TRC files, one per trial
        subjects.csv      # Subject metadata (group, mass, limb lengths)
        ik/               # IK results (generated)
        id/               # ID results (generated)
        figures/          # Output figures (generated)

The scaling step uses rathindlimb.scale.scale_opensim_model() which:
  - Computes segment scale factors from femur/tibia length ratios
  - Runs the OpenSim Scale Tool via osimpy
  - Overrides inertial properties with Hicks regression equations
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

# Path to the generic scale setup XML in rat-hindlimb-model
SCALE_XML_DIR = Path("models/osim/xml")


# =========================================================================
# Step 1: Load subject metadata
# =========================================================================
def load_subjects(csv_path: Path) -> pl.DataFrame:
    """Load subject metadata from subjects.csv.

    Expected columns:
        Subject, Group, Mass, RFemurLength, RTibiaLength, RFootLength,
        LFemurLength, LTibiaLength, LFootLength
    """
    if not csv_path.exists():
        logger.warning(f"Subjects file not found: {csv_path}")
        return pl.DataFrame()
    return pl.read_csv(csv_path)


# =========================================================================
# Step 2: Scale model to subject anthropometrics
# =========================================================================
def scale_model(
    base_model_path: Path,
    subject_name: str,
    parameters: dict,
    trc_path: Path,
    output_dir: Path,
    marker_set_path: Path | None = None,
    generic_setup_path: Path | None = None,
    initial_time: float = 0.0,
    final_time: float | None = None,
) -> Path:
    """Scale the bilateral rat model to subject-specific anthropometrics.

    This uses rathindlimb.scale.scale_opensim_model() which:
      1. Computes manual scale factors from femur/tibia length ratios
         against the base model dimensions
      2. Runs the OpenSim Scale Tool via osimpy.ScaleSettings
      3. Overrides segment inertial properties (mass, COM, MOI)
         using Hicks regression equations

    Parameters are typically read from a C3D file's PROCESSING metadata
    or from a subjects.csv row. Required keys:
        Mass, RFemurLength, RTibiaLength, RFootLength,
        LFemurLength, LTibiaLength, LFootLength
    """
    from rathindlimb.scale import scale_opensim_model, RatScalingParameters

    params = RatScalingParameters(
        Mass=parameters["Mass"],
        RFemurLength=parameters["RFemurLength"],
        RTibiaLength=parameters["RTibiaLength"],
        LFemurLength=parameters["LFemurLength"],
        LTibiaLength=parameters["LTibiaLength"],
        RFootLength=parameters.get("RFootLength", 0.0),
        LFootLength=parameters.get("LFootLength", 0.0),
    )

    # The scale_opensim_model function handles both the OpenSim scale tool
    # and the Hicks regression override internally.
    scale_opensim_model(
        name=subject_name,
        trc_file_name=str(trc_path),
        parameters=params,
        output_dir=str(output_dir),
        initial_time=initial_time,
        final_time=final_time,
        scaled_model_name=f"{subject_name}_scaled.osim",
    )

    scaled_path = output_dir / f"{subject_name}_scaled.osim"
    if not scaled_path.exists():
        raise RuntimeError(f"Scaling failed: {scaled_path} not created")
    logger.info(f"Scaled model -> {scaled_path}")
    return scaled_path


def scale_from_c3d(
    c3d_path: Path,
    base_model_path: Path,
    output_dir: Path,
) -> Path:
    """Scale model using anthropometrics embedded in a C3D static trial.

    Reads PROCESSING parameters from the C3D file (Mass, femur/tibia lengths)
    and passes them to the rathindlimb scaling pipeline.
    """
    from rathindlimb.scale import scaling_parameters_from_c3d, scale_opensim_model

    params = scaling_parameters_from_c3d(str(c3d_path))
    subject_name = c3d_path.stem

    scale_opensim_model(
        name=subject_name,
        trc_file_name=str(c3d_path),
        parameters=params,
        output_dir=str(output_dir),
    )

    scaled_path = output_dir / f"{subject_name}_scaled.osim"
    logger.info(f"Scaled model from C3D -> {scaled_path}")
    return scaled_path


# =========================================================================
# Step 3: Run Inverse Kinematics
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
    logger.info(f"IK complete -> {result.motion_file}")
    return result.motion_file


# =========================================================================
# Step 4: Run Inverse Dynamics
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
    logger.info(f"ID complete -> {result.moments_file}")
    return result.moments_file


# =========================================================================
# Step 5: Load and interpolate results for plotting
# =========================================================================
def load_and_interp_results(
    ik_dir: Path,
    id_dir: Path,
    subject_name: str,
    gait_pct: np.ndarray,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IK/ID results and interpolate to common gait percentages.

    Returns (ik_df, id_df) with a 'gait_percentage' column.  Returns None
    for results that don't exist (file not found or failed).
    """
    from osimpy import sto_to_df

    ik_path = ik_dir / f"{subject_name}_ik_results.mot"
    id_path = id_dir / f"{subject_name}_id_results.sto"

    ik_df = None
    if ik_path.exists():
        ik_df, _ = sto_to_df(str(ik_path))
        # TODO: interpolate to gait_pct

    id_df = None
    if id_path.exists():
        id_df, _ = sto_to_df(str(id_path))
        # TODO: interpolate to gait_pct

    return ik_df, id_df


# =========================================================================
# Step 6: Generate comparison plots
# =========================================================================
def plot_group_comparison(
    group_name: str,
    group_ik: dict[str, list[pl.DataFrame]],
    group_id: dict[str, list[pl.DataFrame]],
    output_dir: Path,
) -> None:
    """Generate kinematics and kinetics comparison plots for a group.

    Figures are saved as {group}_kinematics.png and {group}_kinetics.png
    matching the naming convention used in the manuscript.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")

    # Kinematics plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"{group_name} — Joint Kinematics")
    # TODO: plot mean ± SD across subjects for each coordinate
    #       highlight SPM-significant regions vs Control and vs NR
    plt.tight_layout()
    fig.savefig(
        output_dir / f"{group_name.lower().replace('+', '_')}_kinematics.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # Kinetics plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"{group_name} — Joint Moments")
    # TODO: plot mean ± SD for each moment, SPM highlights
    plt.tight_layout()
    fig.savefig(
        output_dir / f"{group_name.lower().replace('+', '_')}_kinetics.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    logger.info(f"Generated figures for {group_name}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Rat VML analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline from subjects.csv\n"
            "  python scripts/run_analysis.py --data-dir data/vml --model models/bilateral.osim\n"
            "\n"
            "  # Regenerate figures from cached results\n"
            "  python scripts/run_analysis.py --skip-ik --skip-id\n"
            "\n"
            "  # Scale a single subject from a C3D static trial\n"
            "  python scripts/run_analysis.py --c3d data/raw/BAA01_static.c3d\n"
        ),
    )
    parser.add_argument("--data-dir", type=Path, help="Directory with raw/, subjects.csv")
    parser.add_argument("--model", type=Path, help="Path to bilateral rat model (.osim)")
    parser.add_argument("--group", choices=TREATMENT_GROUPS, help="Process only one group")
    parser.add_argument("--c3d", type=Path, help="Scale model from a single C3D static trial")
    parser.add_argument("--skip-ik", action="store_true", help="Skip IK (use cached)")
    parser.add_argument("--skip-id", action="store_true", help="Skip ID (use cached)")
    args = parser.parse_args()

    # Single-subject scaling from C3D
    if args.c3d:
        model_path = args.model or PROJECT_ROOT / "models" / "rat_hindlimb_bilateral.osim"
        output_dir = PROJECT_ROOT / "data" / "scaled_models"
        scale_from_c3d(args.c3d, model_path, output_dir)
        return

    # Multi-subject pipeline
    data_dir = args.data_dir or PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"
    ik_dir = data_dir / "ik"
    id_dir = data_dir / "id"
    figures_dir = data_dir / "figures"
    scaled_dir = data_dir / "scaled_models"

    for d in [ik_dir, id_dir, figures_dir, scaled_dir]:
        d.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(data_dir / "subjects.csv")
    if subjects.is_empty():
        logger.info(
            "No subjects.csv found. "
            "Create one or use --c3d for single-subject scaling."
        )
        return

    model_path = args.model
    if model_path is None:
        # Look for a bilateral model in common locations
        candidates = [
            PROJECT_ROOT / "models" / "rat_hindlimb_bilateral.osim",
            PROJECT_ROOT.parent / "rat-hindlimb-model" / "models" / "osim" / "rat_hindlimb_bilateral.osim",
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
    if model_path is None or not model_path.exists():
        logger.error("Model not found; use --model to specify path")
        return

    logger.info(
        f"Pipeline: {len(subjects)} subjects, model={model_path.name}, "
        f"groups={subjects['Group'].n_unique() if 'Group' in subjects.columns else 'N/A'}"
    )
    logger.info("Run with --skip-ik --skip-id to regenerate figures from cached results")


if __name__ == "__main__":
    main()
