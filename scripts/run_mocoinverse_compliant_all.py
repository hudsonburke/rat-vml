import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hudson/rat-hindlimb-model")
sys.path.insert(0, "/home/hudson/osimpy/src")

import numpy as np
import opensim as osim

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
DEFAULT_OUTPUT_DIR = BASELINE_DIR / (
    "MocoInverse_compliant_all" if TRIAL == "Walk05" else f"MocoInverse_compliant_all_{TRIAL}"
)
OUTPUT_DIR = Path(os.getenv("MOCO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
DEFAULT_OUTPUT_SOLUTION_PATH = OUTPUT_DIR / (
    "moco_inverse_solution_compliant_all.sto"
    if TRIAL == "Walk05"
    else f"moco_inverse_solution_{TRIAL}_compliant_all.sto"
)
OUTPUT_SOLUTION_PATH = Path(
    os.getenv("MOCO_OUTPUT_SOLUTION_PATH", str(DEFAULT_OUTPUT_SOLUTION_PATH))
)
RIGID_INITIAL_GUESS = Path(
    os.getenv(
        "MOCO_INITIAL_GUESS_PATH",
        str(BASELINE_DIR / f"MocoInverse_{TRIAL}" / f"moco_inverse_{TRIAL}.sto"),
    )
)
ABS_FP_MOT = (BASELINE_DIR / f"{TRIAL_TAG}_FP.mot").resolve()
SO_FORCE_GUESS_FILE = (
    BASELINE_DIR / "SO_physiology" / "scaled_scaled_StaticOptimization_force.sto"
)

T_INIT, T_FINAL = TRIAL_WINDOWS[TRIAL]
IK_FILTER_CUTOFF_HZ = 15.0
GENERAL_RESERVE_WEIGHT = 10.0
SACRUM_TRANSLATION_RESERVE_WEIGHT = 1.0
INITIAL_NORMALIZED_TENDON_FORCE = 0.1
MESH_INTERVAL = 0.02
IMPLICIT_AUX_DERIV_WEIGHT = 0.01


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

    print("=" * 60)
    print("Compliant-tendon MocoInverse — all muscles compliant")
    print("=" * 60)
    print(f"  Trial: {TRIAL}")
    print(f"  Time window: {T_INIT:.3f} -> {T_FINAL:.3f} s")

    print("\n[1] Building model processor...")
    model_proc = osim.ModelProcessor(str(MOCO_MODEL))
    model_proc.append(osim.ModOpAddExternalLoads(str(FP_SETUP)))
    model_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_proc.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    model_proc.append(osim.ModOpAddReserves(0.1))
    print("  ✓ All-compliant model processor configured")

    print("\n[2] Building kinematics reference...")
    kinematics = osim.TableProcessor(str(IK_FILE))
    kinematics.append(osim.TabOpLowPassFilter(IK_FILTER_CUTOFF_HZ))
    kinematics.append(osim.TabOpConvertDegreesToRadians())
    kinematics.append(osim.TabOpUseAbsoluteStateNames())
    print(
        "  ✓ Kinematics: TabOpLowPassFilter(15 Hz) + "
        "TabOpConvertDegreesToRadians + TabOpUseAbsoluteStateNames"
    )

    print("\n[3] Configuring MocoInverse...")
    inverse = osim.MocoInverse()
    inverse.setName(f"rat_hindlimb_inverse_compliant_all_{TRIAL}")
    inverse.setModel(model_proc)
    inverse.setKinematics(kinematics)
    inverse.set_initial_time(T_INIT)
    inverse.set_final_time(T_FINAL)
    inverse.set_mesh_interval(MESH_INTERVAL)
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_convergence_tolerance(1e-3)
    inverse.set_constraint_tolerance(1e-4)
    inverse.set_reserves_weight(GENERAL_RESERVE_WEIGHT)
    print(f"  ✓ Mesh interval: {MESH_INTERVAL:.3f} s")

    prev_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)
    t0 = time.time()

    try:
        study = inverse.initialize()
        problem = study.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("excitation_effort"))
        if effort:
            effort.setWeightForControlPattern(".*reserve.*", GENERAL_RESERVE_WEIGHT)
            effort.setWeightForControlPattern(
                ".*reserve_jointset_ground_spine_sacrum_[xyz].*",
                SACRUM_TRANSLATION_RESERVE_WEIGHT,
            )

        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        solver.resetProblem(problem)
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        solver.set_implicit_auxiliary_derivatives_weight(IMPLICIT_AUX_DERIV_WEIGHT)
        solver.set_minimize_implicit_multibody_accelerations(True)
        solver.set_implicit_multibody_accelerations_weight(0.1)
        apply_initial_guess(solver, model_proc, RIGID_INITIAL_GUESS)

        print("\n[4] Running all-compliant inverse solve...")
        solution = study.solve()
        elapsed = time.time() - t0

        if solution.isSealed():
            solution.unseal()

        sol_path = OUTPUT_SOLUTION_PATH
        solution.write(str(sol_path))
        print(f"\n  ✓ All-compliant MocoInverse COMPLETED in {elapsed:.1f}s")
        print(f"    Solution: {sol_path}")
        print(f"    Objective: {solution.getObjective():.6f}")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ All-compliant MocoInverse FAILED after {elapsed:.1f}s: {e}")
        return False
    finally:
        os.chdir(prev_dir)


def apply_initial_guess(solver, model_proc, guess_path):
    if not guess_path.exists():
        print(f"    No rigid initial guess found: {guess_path}")
        return

    print(f"    Loading rigid warm start from: {guess_path}")
    initial_guess = osim.MocoTrajectory(str(guess_path))
    guess = solver.createGuess()
    guess.insertStatesTrajectory(initial_guess.exportToStatesTable(), True)
    guess.insertControlsTrajectory(initial_guess.exportToControlsTable(), True)

    model = model_proc.process()
    model.initSystem()
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


def load_so_normalized_tendon_force_guess(so_force_file, guess_times, muscles):
    if not so_force_file.exists():
        print(f"    SO force guess file not found: {so_force_file}")
        return {}

    column_names, data_rows = _read_storage_file(so_force_file)
    if len(column_names) < 2 or data_rows.size == 0:
        print(f"    SO force guess file is empty or malformed: {so_force_file}")
        return {}

    so_times = data_rows[:, 0]
    force_by_name = {
        name: data_rows[:, idx] for idx, name in enumerate(column_names[1:], start=1)
    }

    tendon_force_map = {}
    for imusc in range(muscles.getSize()):
        muscle = muscles.get(imusc)
        short_name = muscle.getName()
        so_force = force_by_name.get(short_name)
        if so_force is None:
            continue

        max_isometric_force = muscle.getMaxIsometricForce()
        if max_isometric_force <= 0:
            continue

        normalized = np.interp(
            guess_times,
            so_times,
            so_force,
            left=so_force[0],
            right=so_force[-1],
        ) / max_isometric_force
        clipped_normalized = np.clip(normalized, 0.0, 5.0)

        tendon_vector = osim.Vector(len(guess_times), 0.0)
        for itime, value in enumerate(clipped_normalized):
            tendon_vector.set(itime, float(value))
        tendon_force_map[short_name] = tendon_vector

    print(
        "    Loaded SO-informed tendon guesses from "
        f"{so_force_file.name} for {len(tendon_force_map)} muscles"
    )
    return tendon_force_map


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
