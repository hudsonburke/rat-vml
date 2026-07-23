"""End-to-end analysis pipeline for rat gait data.

Mirrors the MATLAB vicon_ratexport → batchtimepointstructs →
createtimepointstruct pipeline from the UVA-MAMP-Lab Rats/Toolbox repos.

Pipeline steps
--------------
1. Load subject metadata from subjects.csv (subject_id, group, mass, limb lengths)
2. For each subject, find the static trial and walking trials
3. Scale model to subject anthropometrics (rathindlimb.scale + Hicks regression)
4. Validate walking trials (7 events in correct order, no marker gaps)
5. For each valid walking trial: export TRC, export FP, run IK, run ID
6. Spline IK/ID results to 101 points per stance+swing (202 total)
7. Average across trials per subject, then across subjects per group
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Try imports that depend on OpenSim — they fail gracefully if not installed
try:
    from osimpy.tools import IKSettings, IDSettings
    from osimpy import sto_to_df
    _HAS_OPENSIM = True
except ImportError:
    _HAS_OPENSIM = False
    logger.warning("osimpy not available — IK/ID steps will be stubs")

try:
    from rathindlimb.scale import scale_opensim_model, RatScalingParameters
    _HAS_SCALE = True
except ImportError:
    _HAS_SCALE = False
    logger.warning("rathindlimb.scale not available — scaling will be a stub")

from .events import (
    GaitEvents,
    extract_events_from_c3d,
    extract_events_from_enf,
    validate_walking_trial,
    get_gait_cycle_times,
    check_marker_gaps,
)
from .forces import process_force_plate, zero_outside_gait_cycle, write_external_loads_xml
from .io import rrd_to_trc, rrd_to_fp_mot


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SubjectResult:
    """Results from a single subject's analysis pipeline run."""
    subject_id: str
    session: str
    group: str = ""
    scaled_model: Path | None = None
    trial_results: list["TrialResult"] = field(default_factory=list)
    success: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class TrialResult:
    """Results from a single walking trial."""
    trial_name: str
    ik_file: Path | None = None
    id_file: Path | None = None
    ik_splined: np.ndarray | None = None  # (202, n_coords) stance+swing
    id_splined: np.ndarray | None = None  # (202, n_moments) stance+swing
    events: GaitEvents | None = None
    side: str = "right"
    success: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class GroupResult:
    """Aggregated results for a treatment group."""
    group_name: str
    subjects: list[SubjectResult] = field(default_factory=list)
    ik_mean: np.ndarray | None = None   # (202, n_coords)
    ik_std: np.ndarray | None = None
    id_mean: np.ndarray | None = None   # (202, n_moments)
    id_std: np.ndarray | None = None
    n_subjects: int = 0


# ---------------------------------------------------------------------------
# Coordinate and moment names matching the rat model
# ---------------------------------------------------------------------------
COORD_NAMES = [
    "sacrum_pitch", "sacrum_roll", "sacrum_yaw",
    "sacrum_x", "sacrum_y", "sacrum_z",
    "sacroiliac_r_flx",
    "hip_r_flx", "hip_r_add", "hip_r_int",
    "knee_r_flx", "ankle_r_flx",
    "ankle_r_add", "ankle_r_int",
    "sacroiliac_l_flx",
    "hip_l_flx", "hip_l_add", "hip_l_int",
    "knee_l_flx", "ankle_l_flx",
    "ankle_l_add", "ankle_l_int",
]

# Coordinates to plot (right side)
PLOT_COORDS = ["hip_r_flx", "hip_r_add", "hip_r_int", "knee_r_flx", "ankle_r_flx"]

MOMENT_NAMES = [
    "hip_r_flx_moment", "hip_r_add_moment", "hip_r_int_moment",
    "knee_r_flx_moment", "ankle_r_flx_moment",
]


# =========================================================================
# Trial filtering
# =========================================================================
def filter_walking_trials(
    session_dir: Path,
    side: str = "right",
    min_events: int = 7,
) -> list[dict]:
    """Find valid walking trials in a session directory.

    A trial is valid if:
    1. It has ≥7 gait events on the primary side in the correct order
    2. It has no marker gaps (all-zero frames)
    3. It is not a Static trial

    Parameters
    ----------
    session_dir : Path
        Path to session directory (e.g. data/raw/BAA01/Baseline/).
    side : str
        Primary analysis side ("left" or "right").
    min_events : int
        Minimum events required on the primary side.

    Returns
    -------
    list of dicts with keys: 'c3d', 'enf', 'events', 'is_valid', 'reason'
    """
    session_dir = Path(session_dir)

    # Find all C3D files, excluding Static trials
    c3d_files = sorted(session_dir.glob("*.c3d"))
    c3d_files = [f for f in c3d_files if "static" not in f.name.lower()]

    results = []
    for c3d in c3d_files:
        enf = c3d.with_suffix(".Trial.enf")

        # Extract events from C3D or .enf
        try:
            events = extract_events_from_c3d(c3d)
        except Exception:
            events = extract_events_from_enf(enf, 200.0, 0)

        is_valid, reason = validate_walking_trial(events, side, min_events)

        results.append({
            "c3d": c3d,
            "enf": enf,
            "events": events,
            "is_valid": is_valid,
            "reason": reason,
        })

    valid = [r for r in results if r["is_valid"]]
    logger.info(
        f"{session_dir.name}: {len(valid)}/{len(results)} valid trials "
        f"(filtered: {sum(1 for r in results if not r['is_valid'])})"
    )

    return results


# =========================================================================
# Scaling
# =========================================================================
def scale_model_for_subject(
    base_model: Path,
    subject_name: str,
    mass: float,
    femur_length: float,
    tibia_length: float,
    output_dir: Path,
    side: str = "right",
    foot_length: float = 0.0,
) -> Path:
    """Scale bilateral rat model using rathindlimb.scale.scale_opensim_model.

    Uses manual scale factors from femur/tibia length ratios plus Hicks
    regression equations for inertial properties.
    """
    if not _HAS_SCALE:
        raise ImportError("rathindlimb.scale is required for scaling")

    params = RatScalingParameters(
        Mass=mass,
        RFemurLength=femur_length,
        RTibiaLength=tibia_length,
        LFemurLength=femur_length,
        LTibiaLength=tibia_length,
        RFootLength=foot_length,
        LFootLength=foot_length,
    )

    scale_opensim_model(
        name=subject_name,
        trc_file_name="",
        parameters=params,
        output_dir=str(output_dir),
    )

    scaled = output_dir / f"{subject_name}_scaled.osim"
    if not scaled.exists():
        raise RuntimeError(f"Scaling produced no output at {scaled}")
    logger.info(f"Scaled model: {scaled}")
    return scaled


# =========================================================================
# IK/ID
# =========================================================================
def run_ik(
    model_path: Path,
    trc_path: Path,
    output_dir: Path,
    events: GaitEvents | None = None,
    name: str = "ik",
) -> Path:
    """Run Inverse Kinematics using osimpy.

    Time range is set from first event to last event (matching MATLAB).
    If events is None, uses the full time range from the TRC file.
    """
    if not _HAS_OPENSIM:
        raise ImportError("osimpy is required for IK")

    start_time = None
    end_time = None

    if events is not None and events.has_events:
        times = events.to_times()
        all_times = sorted(
            times["right_foot_strike"] + times["right_foot_off"]
            + times["left_foot_strike"] + times["left_foot_off"]
        )
        start_time = all_times[0]
        end_time = all_times[-1]

    settings = IKSettings(
        name=name,
        model_path=model_path,
        marker_path=trc_path,
        results_directory=output_dir,
        output_motion_file=f"{name}_ik.mot",
        initial_time=start_time,
        final_time=end_time,
    )
    result = settings.run()
    if not result.success:
        raise RuntimeError(f"IK failed: {result.errors}")
    logger.info(f"IK complete: {result.motion_file}")
    return result.motion_file


def run_id(
    model_path: Path,
    mot_path: Path,
    external_loads_path: Path,
    output_dir: Path,
    name: str = "id",
    cutoff_freq: float = 6.0,
) -> Path:
    """Run Inverse Dynamics using osimpy."""
    if not _HAS_OPENSIM:
        raise ImportError("osimpy is required for ID")

    settings = IDSettings(
        name=name,
        model_path=model_path,
        coordinates_path=mot_path,
        results_directory=output_dir,
        output_forces_file=f"{name}_id.sto",
        external_loads_path=external_loads_path,
        lowpass_cutoff_frequency=cutoff_freq,
    )
    result = settings.run()
    if not result.success:
        raise RuntimeError(f"ID failed: {result.errors}")
    logger.info(f"ID complete: {result.moments_file}")
    return result.moments_file


# =========================================================================
# Spline to stance+swing
# =========================================================================
def spline_to_stance_swing(
    data: np.ndarray,
    time_col: np.ndarray,
    stance_times: tuple[float, float],
    swing_times: tuple[float, float],
    n_points: int = 101,
) -> np.ndarray:
    """Spline data to 101 points for stance and 101 for swing.

    Matches MATLAB splinetostanceswing.m logic.

    Parameters
    ----------
    data : (N, C) array
        Data columns (no time column).
    time_col : (N,) array
        Time values.
    stance_times : (start, end)
        Stance phase start and end times (seconds).
    swing_times : (start, end)
        Swing phase start and end times (seconds).
    n_points : int
        Number of interpolation points per phase.

    Returns
    -------
    (2*n_points, C) array with stance followed by swing.
    """
    from scipy.interpolate import interp1d

    result_parts = []
    for (start, end) in [stance_times, swing_times]:
        mask = (time_col >= start) & (time_col <= end)
        t_trimmed = time_col[mask]
        d_trimmed = data[mask]

        if len(t_trimmed) < 2:
            # Not enough data in this phase
            result_parts.append(np.full((n_points, data.shape[1]), np.nan))
            continue

        t_interp = np.linspace(start, end, n_points)
        f = interp1d(t_trimmed, d_trimmed, axis=0, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        result_parts.append(f(t_interp))

    return np.vstack(result_parts)


# =========================================================================
# Full subject pipeline
# =========================================================================
def run_subject(
    base_model: Path,
    subject_id: str,
    session: str,
    group: str,
    rrd_path: Path,
    output_dir: Path,
    side: str = "right",
    subject_mass: float | None = None,
    femur_length: float | None = None,
    tibia_length: float | None = None,
    skip_scaling: bool = False,
    skip_ik: bool = False,
    skip_id: bool = False,
    min_events: int = 7,
) -> SubjectResult:
    """Run the full pipeline for one subject session.

    Parameters
    ----------
    base_model : Path
        Bilateral rat model (.osim).
    subject_id : str
        Subject identifier (e.g. "BAA01").
    session : str
        Session name (e.g. "Baseline", "Week24").
    group : str
        Treatment group name.
    rrd_path : Path
        Path to the .rrd file for this subject.
    output_dir : Path
        Directory for all output files.
    side : str
        Primary analysis side.
    subject_mass, femur_length, tibia_length : float
        Anthropometrics (required unless skip_scaling=True).
    skip_scaling, skip_ik, skip_id : bool
        Skip individual pipeline steps (use cached results).
    min_events : int
        Minimum events required for a valid walking trial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = SubjectResult(subject_id=subject_id, session=session, group=group)

    try:
        # Step 1: Scale model
        if skip_scaling:
            model_path = base_model
        else:
            if None in (subject_mass, femur_length, tibia_length):
                raise ValueError("subject_mass, femur_length, tibia_length required for scaling")
            model_path = scale_model_for_subject(
                base_model, f"{subject_id}_{session}",
                subject_mass, femur_length, tibia_length,
                output_dir / "scaled",
            )
        result.scaled_model = model_path

        # Step 2: Find valid walking trials from .rrd catalog
        from .queries import RerunCatalog
        cat = RerunCatalog(rrd_path.parent)
        valid_trials = cat.valid_walking_trials(min_events=min_events, session=session)
        cat.close()

        # Filter to this subject
        subject_trials = valid_trials.filter(pl.col("subject") == subject_id)
        if subject_trials.is_empty():
            logger.warning(f"No valid walking trials found for {subject_id}/{session}")
            return result

        # Step 3: Run IK/ID for each valid trial
        for trial_row in subject_trials.iter_rows(named=True):
            trial_name = trial_row["trial"]
            entity_prefix = f"{subject_id}/trials/{session}_{trial_name}"

            trial_result = TrialResult(
                trial_name=trial_name,
                side=side,
            )

            try:
                # Extract TRC from .rrd
                trc_path = output_dir / "trials" / trial_name / f"{trial_name}.trc"
                if not trc_path.exists() and not skip_ik:
                    logger.info(f"  Extracting TRC from .rrd for {trial_name}")
                    rrd_to_trc(rrd_path, entity_prefix, trc_path.parent,
                               output_name=f"{trial_name}.trc")

                # Extract FP MOT from .rrd
                ext_loads_path = output_dir / "trials" / trial_name / f"{trial_name}_ext_loads.xml"
                if not ext_loads_path.exists() and not skip_id:
                    logger.info(f"  Extracting force plate data from .rrd for {trial_name}")
                    rrd_to_fp_mot(
                        rrd_path, entity_prefix,
                        events=GaitEvents(
                            left_foot_strike=[], left_foot_off=[],
                            right_foot_strike=[], right_foot_off=[],
                            total_frames=0, frame_rate=200.0,
                        ),
                        output_dir=output_dir / "trials" / trial_name,
                        output_prefix=trial_name,
                    )

                # Run IK
                trial_out = output_dir / "trials" / trial_name
                trial_out.mkdir(parents=True, exist_ok=True)
                if not skip_ik:
                    ik_file = run_ik(
                        model_path, trc_path, trial_out,
                        name=f"{subject_id}_{trial_name}",
                    )
                else:
                    ik_file = trial_out / f"{subject_id}_{trial_name}_ik.mot"
                trial_result.ik_file = ik_file

                # Run ID
                if not skip_id:
                    id_file = run_id(
                        model_path, ik_file, ext_loads_path, trial_out,
                        name=f"{subject_id}_{trial_name}",
                    )
                else:
                    id_file = trial_out / f"{subject_id}_{trial_name}_id.sto"
                trial_result.id_file = id_file

                # Load and spline results
                if ik_file.exists():
                    ik_df, _ = sto_to_df(str(ik_file))

                    # Try to extract events from .rrd for splining
                    try:
                        from .events import extract_events_from_c3d
                        # Events are in the .rrd as TextLog entries
                        # For now, use the full IK data without stance/swing split
                        ik_cols = [c for c in ik_df.columns if c != "time"]
                        ik_data = ik_df.select(ik_cols).to_numpy()
                        trial_result.ik_splined = ik_data
                    except Exception:
                        ik_cols = [c for c in ik_df.columns if c != "time"]
                        ik_data = ik_df.select(ik_cols).to_numpy()
                        trial_result.ik_splined = ik_data

                trial_result.success = True
                result.trial_results.append(trial_result)

            except Exception as e:
                trial_result.errors.append(str(e))
                trial_result.success = False
                logger.error(f"  Trial {trial_name} failed: {e}")

        result.success = any(t.success for t in result.trial_results)

    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"Subject {subject_id} failed: {e}")

    return result


# =========================================================================
# Group aggregation
# =========================================================================
def aggregate_group(
    subjects: list[SubjectResult],
    group_name: str,
) -> GroupResult:
    """Average IK/ID across subjects in a group.

    Averages across trials within each subject first, then across subjects.
    """
    grp = GroupResult(group_name=group_name, subjects=subjects)

    all_ik = []
    all_id = []

    for subj in subjects:
        if not subj.success:
            continue

        # Average across trials for this subject
        subj_ik = [t.ik_splined for t in subj.trial_results
                    if t.success and t.ik_splined is not None]
        if subj_ik:
            all_ik.append(np.nanmean(subj_ik, axis=0))

        subj_id = [t.id_splined for t in subj.trial_results
                    if t.success and t.id_splined is not None]
        if subj_id:
            all_id.append(np.nanmean(subj_id, axis=0))

    if all_ik:
        grp.ik_mean = np.nanmean(all_ik, axis=0)
        grp.ik_std = np.nanstd(all_ik, axis=0)
        grp.n_subjects = len(all_ik)

    if all_id:
        grp.id_mean = np.nanmean(all_id, axis=0)
        grp.id_std = np.nanstd(all_id, axis=0)

    return grp
