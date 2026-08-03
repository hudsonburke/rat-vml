"""
MocoInverse — Rat hindlimb model.

Uses the same model, IK, and external loads as MocoTrack v2, but solves the
inverse problem: kinematics are prescribed exactly (via PositionMotion) and
the solver finds muscle activations that are consistent with the prescribed
motion while minimising control effort.

Key differences from MocoTrack v2:
  - Kinematics are prescribed, not tracked as a cost → no tracking weight.
  - No state-tracking objective → the problem is smaller and faster.
  - Uses MocoInverse.setKinematics() instead of MocoTrack.setStatesReference().
  - setKinematics takes coordinate columns (short names), not state paths,
    so TabOpUseAbsoluteStateNames is NOT used here.

Usage:
    cd /path/to/BAA01/Baseline
    python run_mocoinverse.py
"""

import os
import sys
import time
import csv
import re
from pathlib import Path

sys.path.insert(0, "/home/hudson/rat-hindlimb-model")
sys.path.insert(0, "/home/hudson/osimpy/src")

import numpy as np
import opensim as osim


def get_env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def get_env_name_set(name):
    value = os.getenv(name, "").strip()
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def get_env_force_override_map(name):
    value = os.getenv(name, "").strip()
    if not value:
        return {}

    result = {}
    for entry in value.split(","):
        item = entry.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Environment variable {name} entries must be name:value, got {item!r}"
            )
        key, raw_force = item.split(":", 1)
        try:
            result[key.strip()] = float(raw_force)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} force must be float in {item!r}"
            ) from exc
    return result


def validate_requested_muscle_names(requested_names, muscles):
    if not requested_names:
        return
    available_names = {muscles.get(index).getName() for index in range(muscles.getSize())}
    unknown = sorted(requested_names - available_names)
    if unknown:
        raise ValueError(
            "Unknown muscle names in MOCO_RIGID_TENDON_MUSCLE_NAMES: "
            + ", ".join(unknown)
        )


def validate_coordinate_force_overrides(requested_overrides, forces):
    if not requested_overrides:
        return
    available_coordinates = set()
    for index in range(forces.getSize()):
        actuator = osim.CoordinateActuator.safeDownCast(forces.get(index))
        if actuator is None:
            continue
        available_coordinates.add(actuator.getCoordinate().getName())

    unknown = sorted(set(requested_overrides) - available_coordinates)
    if unknown:
        raise ValueError(
            "Unknown coordinate names in MOCO_COORDINATE_RESERVE_OPTIMAL_FORCE: "
            + ", ".join(unknown)
            + ". Use coordinate short names such as sacrum_y or hip_r_flx."
        )

# ── Paths ──
BASELINE_DIR = Path(os.getenv("CMC_BASELINE_DIR", "/home/hudson/Downloads/CMC_Runs/BAA01/Baseline"))
TRIAL_WINDOWS = {
    "Walk05": (2.870, 3.450),
    "Walk06": (2.905, 3.400),
    "Walk08": (2.140, 2.600),
    "Walk11": (9.025, 9.435),
    "Walk14": (5.660, 6.075),
}
TRIAL = os.getenv("MOCO_TRIAL", "Walk05")
if TRIAL not in TRIAL_WINDOWS:
    raise ValueError(
        f"Unsupported MOCO_TRIAL {TRIAL!r}. Expected one of: {', '.join(sorted(TRIAL_WINDOWS))}"
    )
TRIAL_TAG = f"BAA01_Baseline_{TRIAL}"
MOCO_MODEL = BASELINE_DIR / "scaled_moco.osim"
IK_FILE = BASELINE_DIR / f"{TRIAL_TAG}_ik.mot"
RAW_FP_SETUP = BASELINE_DIR / f"{TRIAL_TAG}_fp_setup.xml"
OUTPUT_DIR = BASELINE_DIR / "MocoInverse"
DEFAULT_INITIAL_GUESS = BASELINE_DIR / f"MocoInverse_{TRIAL}" / f"moco_inverse_{TRIAL}.sto"
SHARED_COMPLIANT_STAGE1_GUESS = (
    BASELINE_DIR / "MocoInverse_compliant_all" / "moco_inverse_solution_compliant_all.sto"
)
TRIAL_COMPLIANT_STAGE1_GUESS = Path(
    os.getenv(
        "MOCO_TRIAL_COMPLIANT_GUESS_PATH",
        str(
            BASELINE_DIR
            / f"MocoInverse_compliant_all_{TRIAL}"
            / f"moco_inverse_solution_{TRIAL}_compliant_all.sto"
        ),
    )
)
OVERRIDE_INITIAL_GUESS = os.getenv("MOCO_INITIAL_GUESS_PATH")
SO_FORCE_GUESS_FILE = (
    BASELINE_DIR / "SO_physiology" / "scaled_scaled_StaticOptimization_force.sto"
)
TEMP_MODEL_FILE = BASELINE_DIR / f"scaled_moco_selected_ignore_tendon_{TRIAL}.osim"
ABS_FP_MOT = (BASELINE_DIR / f"{TRIAL_TAG}_FP.mot").resolve()
TSL_COMPARISON_FILE = Path(
    "/home/hudson/rat-hindlimb-model/data/parameters/tsl_comparison.csv"
)
OUTPUT_SOLUTION_PATH = Path(
    os.getenv("MOCO_OUTPUT_SOLUTION_PATH", str(OUTPUT_DIR / "moco_inverse_solution.sto"))
)

T_INIT, T_FINAL = TRIAL_WINDOWS[TRIAL]
IK_FILTER_CUTOFF_HZ = 15.0
GENERAL_RESERVE_WEIGHT = get_env_float("MOCO_GENERAL_RESERVE_WEIGHT", 10.0)
SACRUM_TRANSLATION_RESERVE_WEIGHT = get_env_float(
    "MOCO_SACRUM_TRANSLATION_RESERVE_WEIGHT", 0.01
)
SACRUM_PITCH_RESERVE_WEIGHT = get_env_float("MOCO_SACRUM_PITCH_RESERVE_WEIGHT", 10.0)
SACRUM_ROLL_RESERVE_WEIGHT = get_env_float("MOCO_SACRUM_ROLL_RESERVE_WEIGHT", 10.0)
SACRUM_YAW_RESERVE_WEIGHT = get_env_float("MOCO_SACRUM_YAW_RESERVE_WEIGHT", 10.0)
GENERAL_RESERVE_OPTIMAL_FORCE = get_env_float("MOCO_GENERAL_RESERVE_OPTIMAL_FORCE", 0.1)
SACRUM_XZ_RESERVE_OPTIMAL_FORCE = get_env_float(
    "MOCO_SACRUM_XZ_RESERVE_OPTIMAL_FORCE", 0.1
)
SACRUM_Y_RESERVE_OPTIMAL_FORCE = get_env_float("MOCO_SACRUM_Y_RESERVE_OPTIMAL_FORCE", 1.5)
SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE = get_env_float(
    "MOCO_SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE", 0.1
)
DGF_FIBER_DAMPING = get_env_float("MOCO_DGF_FIBER_DAMPING", 0.1)
DGF_ACTIVE_FORCE_WIDTH_SCALE = get_env_float(
    "MOCO_DGF_ACTIVE_FORCE_WIDTH_SCALE", 1.5
)
INITIAL_NORMALIZED_TENDON_FORCE = get_env_float(
    "MOCO_INITIAL_NORMALIZED_TENDON_FORCE", 0.1
)
MESH_INTERVAL = get_env_float("MOCO_MESH_INTERVAL", 0.01)
IMPLICIT_AUX_DERIV_WEIGHT = get_env_float("MOCO_IMPLICIT_AUX_DERIV_WEIGHT", 0.0)
TSL_IGNORE_THRESHOLD_MM = get_env_float("MOCO_TSL_IGNORE_THRESHOLD_MM", 0.5)
SOLVER_MAX_ITERATIONS = int(get_env_float("MOCO_SOLVER_MAX_ITERATIONS", 3000))
EXTRA_RIGID_TENDON_MUSCLES = get_env_name_set("MOCO_RIGID_TENDON_MUSCLE_NAMES")
COORDINATE_RESERVE_FORCE_OVERRIDES = get_env_force_override_map(
    "MOCO_COORDINATE_RESERVE_OPTIMAL_FORCE"
)
SACRUM_X_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_x"
SACRUM_Y_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_y"
SACRUM_Z_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_z"
SACRUM_ROTATION_RESERVE_NAMES = {
    "reserve_jointset_ground_spine_sacrum_pitch",
    "reserve_jointset_ground_spine_sacrum_roll",
    "reserve_jointset_ground_spine_sacrum_yaw",
}

# Formerly-locked coordinates and their default values (radians)
LOCKED_COORDS = {
    "sacroiliac_r_flx": 0.06457718,  # 3.70°
    "ankle_r_add": -0.08726646,  # -5.00°
    "ankle_r_int": 0.0,  # 0.00°
    "sacroiliac_l_flx": 0.06457718,  # 3.70°
    "ankle_l_add": -0.08726646,  # -5.00°
    "ankle_l_int": 0.0,  # 0.00°
}


def fix_fp_setup_path(trial_tag, raw_fp_setup, baseline_dir):
    if not raw_fp_setup.exists():
        raise FileNotFoundError(f"FP setup file not found for {trial_tag}: {raw_fp_setup}")

    expected_relative = f"{trial_tag}_FP.mot"
    content = raw_fp_setup.read_text()

    match = re.search(r"<datafile>(.*?)</datafile>", content)
    if match and match.group(1).strip() == expected_relative:
        return raw_fp_setup

    fixed_content = re.sub(
        r"<datafile>.*?</datafile>",
        f"<datafile>{expected_relative}</datafile>",
        content,
    )
    fixed_path = baseline_dir / f"{trial_tag}_fp_setup_fixed.xml"
    fixed_path.write_text(fixed_content)
    return fixed_path


FP_SETUP = fix_fp_setup_path(TRIAL_TAG, RAW_FP_SETUP, BASELINE_DIR)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not IK_FILE.exists():
        raise FileNotFoundError(f"IK file not found for {TRIAL}: {IK_FILE}")
    if not FP_SETUP.exists():
        raise FileNotFoundError(f"FP setup file not found for {TRIAL}: {FP_SETUP}")
    if not ABS_FP_MOT.exists():
        raise FileNotFoundError(f"FP motion file not found for {TRIAL}: {ABS_FP_MOT}")

    ignore_tendon_compliance_muscles = load_thresholded_ignore_tendon_compliance_muscles(
        TSL_COMPARISON_FILE, TSL_IGNORE_THRESHOLD_MM
    )
    ignore_tendon_compliance_muscles.update(EXTRA_RIGID_TENDON_MUSCLES)

    print("=" * 60)
    print("MocoInverse — Rat Hindlimb Model")
    print("=" * 60)
    print(f"  Trial: {TRIAL}")
    print(f"  Time window: {T_INIT:.3f} -> {T_FINAL:.3f} s")

    # ── Model Processor ──
    print("\n[1] Building model processor...")
    model_proc = build_model_processor_with_selected_rigid_tendons(
        ignore_tendon_compliance_muscles
    )

    print("  ✓ Model processor configured")
    print(f"    - General reserve optimal_force: {GENERAL_RESERVE_OPTIMAL_FORCE}")
    print(
        "    - Sacrum_y reserve optimal_force: "
        f"{SACRUM_Y_RESERVE_OPTIMAL_FORCE}"
    )
    print(
        "    - Sacrum_x/z reserve optimal_force: "
        f"{SACRUM_XZ_RESERVE_OPTIMAL_FORCE}"
    )
    print(
        "    - Sacrum rotation reserve optimal_force: "
        f"{SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE}"
    )
    print(f"    - DGF fiber damping: {DGF_FIBER_DAMPING}")
    print(f"    - DGF active force-width scale: {DGF_ACTIVE_FORCE_WIDTH_SCALE}")
    print(
        "    - Sacrum translation reserve weight: "
        f"{SACRUM_TRANSLATION_RESERVE_WEIGHT}"
    )
    print(
        "    - Sacrum rotation reserve weight: "
        f"pitch={SACRUM_PITCH_RESERVE_WEIGHT}, "
        f"roll={SACRUM_ROLL_RESERVE_WEIGHT}, yaw={SACRUM_YAW_RESERVE_WEIGHT}"
    )
    print(
        "    - Ignore tendon compliance threshold: "
        f"Walk TSL <= {TSL_IGNORE_THRESHOLD_MM:.3f} mm"
    )
    print(
        "    - Ignore tendon compliance for: "
        + ", ".join(sorted(ignore_tendon_compliance_muscles))
    )
    print(
        "    - Extra rigid-tendon overrides: "
        + (", ".join(sorted(EXTRA_RIGID_TENDON_MUSCLES)) if EXTRA_RIGID_TENDON_MUSCLES else "none")
    )
    print(
        "    - Coordinate reserve force overrides: "
        + (
            ", ".join(
                f"{name}={force}"
                for name, force in sorted(COORDINATE_RESERVE_FORCE_OVERRIDES.items())
            )
            if COORDINATE_RESERVE_FORCE_OVERRIDES
            else "none"
        )
    )

    # ── Kinematics ──
    print("\n[2] Building kinematics reference...")
    kinematics = osim.TableProcessor(str(IK_FILE))
    kinematics.append(osim.TabOpLowPassFilter(IK_FILTER_CUTOFF_HZ))
    kinematics.append(osim.TabOpConvertDegreesToRadians())
    kinematics.append(osim.TabOpUseAbsoluteStateNames())
    print(
        "  ✓ Kinematics: TabOpLowPassFilter(15 Hz) + "
        "TabOpConvertDegreesToRadians + TabOpUseAbsoluteStateNames"
    )

    # ── Solve ──
    stage_guess_path, tendon_guess_mode = choose_initial_guess_strategy()

    print("\n[3] Running compliant-tendon MocoInverse...")
    print(
        "    Single-stage 10 ms solve with "
        f"{describe_tendon_guess_mode(tendon_guess_mode)}"
    )
    prev_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)
    t0 = time.time()

    try:
        solution = run_inverse_stage(
            model_proc=model_proc,
            kinematics=kinematics,
            stage_name="main",
            mesh_interval=MESH_INTERVAL,
            guess_path=stage_guess_path,
            tendon_guess_mode=tendon_guess_mode,
            output_path=OUTPUT_SOLUTION_PATH,
        )
        elapsed = time.time() - t0

        moco_sol = solution

        if moco_sol.isSealed():
            moco_sol.unseal()

        sol_path = OUTPUT_SOLUTION_PATH

        print(f"\n  ✓ MocoInverse COMPLETED in {elapsed:.1f}s")
        print(f"    Solution: {sol_path}")
        print(f"    Objective: {moco_sol.getObjective():.6f}")

        # Print detailed objective breakdown
        _analyze_solution(moco_sol, sol_path)
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ MocoInverse FAILED after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        os.chdir(prev_dir)


def _analyze_solution(solution, sol_path):
    """Quick analysis of the solution quality."""
    col_names, data = _read_storage_file(sol_path)

    with open(sol_path) as f:
        lines = f.readlines()

    # Parse header for objective breakdown
    print("\n  Solution header:")
    for line in lines:
        if line.strip() == "endheader":
            break
        if "=" in line and not line.startswith("time"):
            key, val = line.strip().split("=", 1)
            print(f"    {key}: {val}")

    # Check reserve magnitudes
    print(f"\n  Reserve actuator check:")
    for ci, cname in enumerate(col_names):
        if "reserve" in cname.lower():
            max_abs = np.max(np.abs(data[:, ci]))
            short = cname.split("/")[-1]
            if max_abs > 0.001:
                print(f"    {short:45s} max|val|={max_abs:.4f}")

    # Check activation range
    act_cols = [
        (ci, cname) for ci, cname in enumerate(col_names) if "/activation" in cname
    ]
    if act_cols:
        act_data = data[:, [c[0] for c in act_cols]]
        print(f"\n  Muscle activation summary ({len(act_cols)} muscles):")
        print(
            f"    min={act_data.min():.4f}  max={act_data.max():.4f}  "
            f"mean={act_data.mean():.4f}  std={act_data.std():.4f}"
        )
        act_range = act_data.max(axis=0) - act_data.min(axis=0)
        print(
            f"    Per-muscle range: min={act_range.min():.4f}  "
            f"max={act_range.max():.4f}  mean={act_range.mean():.4f}"
        )


def choose_initial_guess_strategy():
    if OVERRIDE_INITIAL_GUESS:
        override_path = Path(OVERRIDE_INITIAL_GUESS)
        if override_path.exists():
            print(f"  ✓ Using override initial guess: {override_path}")
            return override_path, "preserve"
        print(f"  ⚠ Override initial guess not found: {override_path}")

    if TRIAL_COMPLIANT_STAGE1_GUESS.exists():
        print(
            "  ✓ Using trial-matched compliant inverse as warm start: "
            f"{TRIAL_COMPLIANT_STAGE1_GUESS}"
        )
        return TRIAL_COMPLIANT_STAGE1_GUESS, "preserve"

    if DEFAULT_INITIAL_GUESS.exists():
        print(f"  ✓ Using trial-matched rigid inverse warm start: {DEFAULT_INITIAL_GUESS}")
        return DEFAULT_INITIAL_GUESS, "so"

    if SHARED_COMPLIANT_STAGE1_GUESS.exists():
        print(
            "  ⚠ Falling back to shared cross-trial compliant inverse warm start: "
            f"{SHARED_COMPLIANT_STAGE1_GUESS}"
        )
        return SHARED_COMPLIANT_STAGE1_GUESS, "preserve"

    print(f"  ⚠ Falling back to rigid inverse warm start: {DEFAULT_INITIAL_GUESS}")
    return DEFAULT_INITIAL_GUESS, "so"


def describe_tendon_guess_mode(mode):
    if mode == "preserve":
        return "preserved tendon states from compliant 20 ms inverse"
    if mode == "so":
        return "SO tendon-force guess"
    return mode


def run_inverse_stage(
    model_proc, kinematics, stage_name, mesh_interval, guess_path, tendon_guess_mode, output_path
):
    print(f"\n[{stage_name}] Configuring MocoInverse...")
    inverse = osim.MocoInverse()
    inverse.setName(f"rat_hindlimb_inverse_{stage_name}")
    inverse.setModel(model_proc)
    inverse.setKinematics(kinematics)
    inverse.set_initial_time(T_INIT)
    inverse.set_final_time(T_FINAL)
    inverse.set_mesh_interval(mesh_interval)
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_convergence_tolerance(1e-3)
    inverse.set_constraint_tolerance(1e-4)
    inverse.set_reserves_weight(GENERAL_RESERVE_WEIGHT)

    print(f"  ✓ {stage_name} mesh interval: {mesh_interval:.3f} s")

    study = inverse.initialize()
    problem = study.updProblem()
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("excitation_effort"))
    if effort:
        effort.setWeightForControlPattern(".*reserve.*", GENERAL_RESERVE_WEIGHT)
        effort.setWeightForControlPattern(
            ".*reserve_jointset_ground_spine_sacrum_[xyz].*",
            SACRUM_TRANSLATION_RESERVE_WEIGHT,
        )
        effort.setWeightForControlPattern(
            ".*reserve_jointset_ground_spine_sacrum_pitch.*",
            SACRUM_PITCH_RESERVE_WEIGHT,
        )
        effort.setWeightForControlPattern(
            ".*reserve_jointset_ground_spine_sacrum_roll.*",
            SACRUM_ROLL_RESERVE_WEIGHT,
        )
        effort.setWeightForControlPattern(
            ".*reserve_jointset_ground_spine_sacrum_yaw.*",
            SACRUM_YAW_RESERVE_WEIGHT,
        )

    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.resetProblem(problem)
    solver.set_optim_max_iterations(SOLVER_MAX_ITERATIONS)
    solver.set_minimize_implicit_auxiliary_derivatives(True)
    solver.set_implicit_auxiliary_derivatives_weight(IMPLICIT_AUX_DERIV_WEIGHT)
    solver.set_minimize_implicit_multibody_accelerations(True)
    solver.set_implicit_multibody_accelerations_weight(0.1)
    print(f"  Solver max iterations: {SOLVER_MAX_ITERATIONS}")
    apply_initial_guess(
        solver=solver,
        model_proc=model_proc,
        guess_path=guess_path,
        tendon_guess_mode=tendon_guess_mode,
    )

    print(f"  Running {stage_name} solve...")
    solution = study.solve()
    if solution.isSealed():
        solution.unseal()
    solution.write(str(output_path))
    print(f"  ✓ {stage_name} solution written to: {output_path}")
    return solution


def apply_initial_guess(solver, model_proc, guess_path, tendon_guess_mode):
    if not guess_path.exists():
        print(f"    No initial guess found for {tendon_guess_mode}: {guess_path}")
        return

    print(f"    Loading initial guess from: {guess_path}")
    initial_guess = osim.MocoTrajectory(str(guess_path))
    guess = solver.createGuess()
    model = model_proc.process()
    model.initSystem()
    allowed_state_names = get_model_state_names(model)
    filtered_states_table, removed_state_names = filter_states_table(
        initial_guess.exportToStatesTable(), allowed_state_names
    )
    if removed_state_names:
        print(
            "    Removed state columns not present in current model: "
            f"{len(removed_state_names)}"
        )
        print("      " + ", ".join(sorted(removed_state_names)))
    guess.insertStatesTrajectory(filtered_states_table, True)
    guess.insertControlsTrajectory(initial_guess.exportToControlsTable(), True)

    if tendon_guess_mode != "so":
        print("    Preserving tendon states from compliant-tendon trajectory guess")
        solver.setGuess(guess)
        return

    muscles = model.getMuscles()
    guess_times = np.array([guess.getTime()[i] for i in range(guess.getNumTimes())])
    tendon_force_map = load_so_normalized_tendon_force_guess(
        SO_FORCE_GUESS_FILE, guess_times, muscles
    )
    n_times = guess.getNumTimes()
    default_tendon_force_guess = osim.Vector(n_times, INITIAL_NORMALIZED_TENDON_FORCE)
    initialized = 0
    fallback = 0
    for imusc in range(muscles.getSize()):
        muscle = muscles.get(imusc)
        state_name = muscle.getAbsolutePathString() + "/normalized_tendon_force"
        short_name = muscle.getName()
        tendon_force_guess = tendon_force_map.get(short_name, default_tendon_force_guess)
        try:
            guess.setState(state_name, tendon_force_guess)
            initialized += 1
            if short_name not in tendon_force_map:
                fallback += 1
        except Exception:
            pass

    print(
        "    Initialized normalized_tendon_force guess for "
        f"{initialized} muscles using SO-informed tendon forces "
        f"({fallback} fallback to {INITIAL_NORMALIZED_TENDON_FORCE})"
    )
    solver.setGuess(guess)


def build_model_processor_with_selected_rigid_tendons(ignore_tendon_compliance_muscles):
    base_proc = osim.ModelProcessor(str(MOCO_MODEL))
    base_proc.append(osim.ModOpAddExternalLoads(str(FP_SETUP)))
    base_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    base_proc.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())
    base_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    base_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(DGF_ACTIVE_FORCE_WIDTH_SCALE))
    base_proc.append(osim.ModOpAddReserves(GENERAL_RESERVE_OPTIMAL_FORCE))

    model = base_proc.process()
    model.initSystem()
    muscles = model.updMuscles()
    forces = model.updForceSet()
    validate_requested_muscle_names(ignore_tendon_compliance_muscles, muscles)
    validate_coordinate_force_overrides(COORDINATE_RESERVE_FORCE_OVERRIDES, forces)
    changed = 0
    for imusc in range(muscles.getSize()):
        muscle = muscles.get(imusc)
        dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
        if dgf is not None:
            dgf.set_fiber_damping(DGF_FIBER_DAMPING)
        if muscle.getName() in ignore_tendon_compliance_muscles:
            if dgf is not None:
                dgf.set_ignore_tendon_compliance(True)
                changed += 1

    sacrum_reserve_changes = 0
    for iforce in range(forces.getSize()):
        force = forces.get(iforce)
        actuator = osim.CoordinateActuator.safeDownCast(force)
        if actuator is None:
            continue

        coordinate = actuator.getCoordinate().getName()
        override_force = COORDINATE_RESERVE_FORCE_OVERRIDES.get(coordinate)
        if override_force is not None:
            actuator.setOptimalForce(override_force)
            sacrum_reserve_changes += 1
            continue

        if force.getName() == SACRUM_Y_RESERVE_NAME:
            actuator.setOptimalForce(SACRUM_Y_RESERVE_OPTIMAL_FORCE)
            sacrum_reserve_changes += 1
        elif force.getName() in {SACRUM_X_RESERVE_NAME, SACRUM_Z_RESERVE_NAME}:
            actuator.setOptimalForce(SACRUM_XZ_RESERVE_OPTIMAL_FORCE)
            sacrum_reserve_changes += 1
        elif force.getName() in SACRUM_ROTATION_RESERVE_NAMES:
            actuator.setOptimalForce(SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE)
            sacrum_reserve_changes += 1

    print(
        "    Selected muscles with ignore_tendon_compliance=True: "
        f"{changed}/{len(ignore_tendon_compliance_muscles)}"
    )
    print(
        "    Sacrum reserve overrides applied: "
        f"{sacrum_reserve_changes}/6"
    )
    model.printToXML(str(TEMP_MODEL_FILE))
    rewrite_external_load_paths(TEMP_MODEL_FILE)
    print(f"    Temporary model written to: {TEMP_MODEL_FILE}")
    return osim.ModelProcessor(str(TEMP_MODEL_FILE))


def rewrite_external_load_paths(model_file):
    text = model_file.read_text()
    text = text.replace(
        f"<datafile>{TRIAL_TAG}_FP.mot</datafile>",
        f"<datafile>{ABS_FP_MOT}</datafile>",
    )
    text = text.replace(
        f"<data_source_name>{TRIAL_TAG}</data_source_name>",
        f"<data_source_name>{ABS_FP_MOT}</data_source_name>",
    )
    text = text.replace(
        f"<data_source_name>{TRIAL_TAG}_fp.mot</data_source_name>",
        f"<data_source_name>{ABS_FP_MOT}</data_source_name>",
    )
    model_file.write_text(text)


def load_so_normalized_tendon_force_guess(so_force_file, guess_times, muscles):
    if not so_force_file.exists():
        print(f"    SO force guess file not found: {so_force_file}")
        return {}

    column_names, data_rows = _read_storage_file(so_force_file)
    if len(column_names) < 2 or data_rows.size == 0:
        print(f"    SO force guess file is empty or malformed: {so_force_file}")
        return {}

    so_times = data_rows[:, 0]
    force_by_name = {name: data_rows[:, idx] for idx, name in enumerate(column_names[1:], start=1)}

    tendon_force_map = {}
    clipped = 0
    missing = 0
    for imusc in range(muscles.getSize()):
        muscle = muscles.get(imusc)
        short_name = muscle.getName()
        so_force = force_by_name.get(short_name)
        if so_force is None:
            missing += 1
            continue

        max_isometric_force = muscle.getMaxIsometricForce()
        if max_isometric_force <= 0:
            missing += 1
            continue

        normalized = np.interp(
            guess_times,
            so_times,
            so_force,
            left=so_force[0],
            right=so_force[-1],
        ) / max_isometric_force
        clipped_normalized = np.clip(normalized, 0.0, 5.0)
        clipped += int(np.count_nonzero(np.abs(clipped_normalized - normalized) > 1e-12))

        tendon_vector = osim.Vector(len(guess_times), 0.0)
        for itime, value in enumerate(clipped_normalized):
            tendon_vector.set(itime, float(value))
        tendon_force_map[short_name] = tendon_vector

    print(
        "    Loaded SO-informed tendon guesses from "
        f"{so_force_file.name} for {len(tendon_force_map)} muscles "
        f"({missing} missing/invalid, {clipped} clipped samples)"
    )
    return tendon_force_map


def load_thresholded_ignore_tendon_compliance_muscles(tsl_csv_path, threshold_mm):
    if not tsl_csv_path.exists():
        raise FileNotFoundError(f"TSL comparison file not found: {tsl_csv_path}")

    ignore_muscles = set()
    with open(tsl_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbrev = row.get("Abbreviation", "").strip()
            if not abbrev:
                continue
            walk_tsl_mm = float(row["Walk TSL (mm)"])
            if walk_tsl_mm <= threshold_mm:
                ignore_muscles.add(f"L_{abbrev}")
                ignore_muscles.add(f"R_{abbrev}")

    return ignore_muscles


def get_model_state_names(model):
    state_names = model.getStateVariableNames()
    return {state_names.get(i) for i in range(state_names.getSize())}


def filter_states_table(states_table, allowed_state_names):
    removed = []
    for label in tuple(states_table.getColumnLabels()):
        if label not in allowed_state_names:
            states_table.removeColumn(label)
            removed.append(label)
    return states_table, removed


def _read_storage_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    header_end = next(
        (i for i, line in enumerate(lines) if line.strip() == "endheader"), None
    )
    if header_end is None or header_end + 1 >= len(lines):
        raise ValueError(f"Malformed storage file: {file_path}")

    column_names = lines[header_end + 1].strip().split("\t")
    data_rows = []
    for line in lines[header_end + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        vals = stripped.split("\t")
        if len(vals) != len(column_names):
            continue
        data_rows.append([float(v) for v in vals])

    return column_names, np.array(data_rows, dtype=float)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
