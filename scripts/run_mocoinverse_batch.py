"""
MocoInverse — Batch runner for BAA01 walking trials.

Runs the MocoInverse configuration sequentially on each trial.
Each trial takes ~3-5 min (much faster than MocoTrack).

Usage:
    python run_mocoinverse_batch.py                    # Run all 4 remaining trials
    python run_mocoinverse_batch.py Walk06 Walk08      # Run specific trials
    python run_mocoinverse_batch.py --include-walk05   # Re-run Walk05 too

Results are saved to MocoInverse_{trial}/ subdirectories.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hudson/rat-hindlimb-model")
sys.path.insert(0, "/home/hudson/osimpy/src")

import opensim as osim

# ── Constants ──
BASELINE_DIR = Path(os.getenv("CMC_BASELINE_DIR", "/home/hudson/Downloads/CMC_Runs/BAA01/Baseline"))
MOCO_MODEL = BASELINE_DIR / "scaled_moco.osim"

# Trial configurations: trial_name -> (t_init, t_final)
TRIALS = {
    "Walk05": (2.870, 3.450),
    "Walk06": (2.905, 3.400),
    "Walk08": (2.140, 2.600),
    "Walk11": (9.025, 9.435),
    "Walk14": (5.660, 6.075),
}
IK_FILTER_CUTOFF_HZ = 15.0
GENERAL_RESERVE_WEIGHT = 10.0
SACRUM_TRANSLATION_RESERVE_WEIGHT = 1.0


def fix_fp_setup(trial: str) -> Path:
    """Ensure the FP setup XML has a relative datafile path."""
    fp_setup = BASELINE_DIR / f"BAA01_Baseline_{trial}_fp_setup.xml"
    fp_data = BASELINE_DIR / f"BAA01_Baseline_{trial}_FP.mot"

    if not fp_setup.exists():
        raise FileNotFoundError(f"FP setup not found: {fp_setup}")
    if not fp_data.exists():
        raise FileNotFoundError(f"FP data not found: {fp_data}")

    content = fp_setup.read_text()
    expected_relative = f"BAA01_Baseline_{trial}_FP.mot"

    match = re.search(r"<datafile>(.*?)</datafile>", content)
    if match and match.group(1).strip() == expected_relative:
        return fp_setup

    fixed_content = re.sub(
        r"<datafile>.*?</datafile>",
        f"<datafile>{expected_relative}</datafile>",
        content,
    )
    fixed_path = BASELINE_DIR / f"BAA01_Baseline_{trial}_fp_setup_fixed.xml"
    fixed_path.write_text(fixed_content)
    return fixed_path


def run_trial(trial: str, t_init: float, t_final: float) -> dict:
    """Run MocoInverse for a single trial. Returns result dict."""

    print("\n" + "=" * 60)
    print(f"MocoInverse — BAA01 Baseline {trial}")
    print(f"  Time range: {t_init:.3f} -> {t_final:.3f} s")
    print("=" * 60)

    ik_file = BASELINE_DIR / f"BAA01_Baseline_{trial}_ik.mot"
    if not ik_file.exists():
        return {"trial": trial, "status": "SKIPPED", "reason": f"IK file not found: {ik_file}"}

    try:
        fp_setup = fix_fp_setup(trial)
    except FileNotFoundError as e:
        return {"trial": trial, "status": "SKIPPED", "reason": str(e)}

    output_dir = BASELINE_DIR / f"MocoInverse_{trial}"
    output_dir.mkdir(exist_ok=True)

    # ── Model Processor ──
    print("\n[1] Building model processor...")
    model_proc = osim.ModelProcessor(str(MOCO_MODEL))
    model_proc.append(osim.ModOpAddExternalLoads(str(fp_setup)))
    model_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_proc.append(osim.ModOpIgnoreTendonCompliance())
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    model_proc.append(osim.ModOpAddReserves(0.1))
    print("  Model processor configured (reserves optimal_force=0.1)")

    # ── Kinematics ──
    print("[2] Building kinematics reference...")
    kinematics = osim.TableProcessor(str(ik_file))
    kinematics.append(osim.TabOpLowPassFilter(IK_FILTER_CUTOFF_HZ))
    kinematics.append(osim.TabOpConvertDegreesToRadians())
    kinematics.append(osim.TabOpUseAbsoluteStateNames())
    print(
        "  Kinematics: TabOpLowPassFilter(15 Hz) + "
        "TabOpConvertDegreesToRadians + TabOpUseAbsoluteStateNames"
    )

    # ── MocoInverse ──
    print("[3] Configuring MocoInverse...")
    inverse = osim.MocoInverse()
    inverse.setName(f"rat_hindlimb_inverse_{trial}")
    inverse.setModel(model_proc)
    inverse.setKinematics(kinematics)
    inverse.set_initial_time(t_init)
    inverse.set_final_time(t_final)
    inverse.set_mesh_interval(0.02)
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_convergence_tolerance(1e-3)
    inverse.set_constraint_tolerance(1e-4)
    inverse.set_reserves_weight(GENERAL_RESERVE_WEIGHT)

    duration_s = t_final - t_init
    n_mesh = int(duration_s / 0.02)
    print(f"  Mesh: 20ms (~{n_mesh} points over {duration_s:.3f}s)")
    print(f"  General reserves weight: {GENERAL_RESERVE_WEIGHT}")
    print(f"  Sacrum translation reserve weight: {SACRUM_TRANSLATION_RESERVE_WEIGHT}")

    # ── Solve ──
    print(f"[4] Solving {trial}... (expect ~3-5 min)")
    prev_dir = os.getcwd()
    os.chdir(output_dir)
    t0 = time.time()

    try:
        study = inverse.initialize()
        problem = study.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal("excitation_effort")
        )
        if effort:
            effort.setWeightForControlPattern(".*reserve.*", GENERAL_RESERVE_WEIGHT)
            effort.setWeightForControlPattern(
                ".*reserve_jointset_ground_spine_sacrum_[xyz].*",
                SACRUM_TRANSLATION_RESERVE_WEIGHT,
            )

        solution = study.solve()
        elapsed = time.time() - t0

        moco_sol = solution
        if moco_sol.isSealed():
            moco_sol.unseal()

        sol_path = output_dir / f"moco_inverse_{trial}.sto"
        moco_sol.write(str(sol_path))

        result = _analyze_solution(moco_sol, sol_path, trial)
        result["elapsed_s"] = elapsed
        result["elapsed_min"] = elapsed / 60

        print(f"\n  {trial} SOLVED in {elapsed/60:.1f} min")
        print(f"    Objective: {result['objective']:.6f}")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  {trial} FAILED after {elapsed/60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        return {"trial": trial, "status": "SOLVE_FAILED", "reason": str(e),
                "elapsed_s": elapsed}
    finally:
        os.chdir(prev_dir)


def _analyze_solution(solution, sol_path, trial):
    """Analyze solution and return result dict."""
    import numpy as np

    result = {"trial": trial, "status": "Solve_Succeeded", "solution_path": str(sol_path)}

    with open(sol_path) as f:
        lines = f.readlines()

    # Parse header
    for line in lines:
        if line.strip() == "endheader":
            break
        if "=" in line and not line.startswith("time"):
            key, val = line.strip().split("=", 1)
            key = key.strip()
            val = val.strip()
            if key == "objective":
                result["objective"] = float(val)
            elif key.startswith("objective_"):
                result[key] = float(val)
            elif key == "num_iterations":
                result["iterations"] = int(val)
            elif key == "solver_duration":
                result["solver_duration_s"] = float(val)
            elif key == "status":
                result["status"] = val

    # Parse data
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = i
            break

    col_names = lines[header_end + 1].strip().split("\t")
    data = []
    for line in lines[header_end + 2:]:
        vals = line.strip().split("\t")
        if len(vals) == len(col_names):
            data.append([float(v) for v in vals])
    data = np.array(data)

    # Reserve analysis
    reserve_max = {}
    for ci, cname in enumerate(col_names):
        if "reserve" in cname.lower():
            max_abs = np.max(np.abs(data[:, ci]))
            short = cname.split("/")[-1]
            if max_abs > 0.001:
                reserve_max[short] = max_abs
    result["max_reserve"] = max(reserve_max.values()) if reserve_max else 0.0
    result["reserves"] = reserve_max

    # Activation analysis
    act_cols = [(ci, cname) for ci, cname in enumerate(col_names) if "/activation" in cname]
    if act_cols:
        act_data = data[:, [c[0] for c in act_cols]]
        result["act_min"] = float(act_data.min())
        result["act_max"] = float(act_data.max())
        result["act_mean"] = float(act_data.mean())

    # Print summary
    print(f"\n  Reserves: max={result.get('max_reserve', 0):.4f}")
    for rname, rval in sorted(reserve_max.items(), key=lambda x: -x[1])[:5]:
        print(f"    {rname:45s} max|val|={rval:.4f}")
    print(f"  Activations: [{result.get('act_min', 0):.3f}, {result.get('act_max', 0):.3f}], mean={result.get('act_mean', 0):.3f}")

    return result


def print_summary(results: list[dict]):
    """Print a summary table of all trial results."""
    print("\n" + "=" * 90)
    print("BATCH SUMMARY — MocoInverse BAA01 Trials")
    print("=" * 90)
    print(f"{'Trial':<10} {'Status':<18} {'Iter':>5} {'Time':>8} {'Objective':>12} "
          f"{'MaxRes':>8} {'ActMean':>8}")
    print("-" * 90)
    for r in results:
        if r["status"] in ("SKIPPED", "INIT_FAILED", "SOLVE_FAILED"):
            print(f"{r['trial']:<10} {r['status']:<18} {'':>5} {'':>8} {'':>12} "
                  f"{'':>8} {'':>8}  {r.get('reason', '')}")
        else:
            mins = r.get("elapsed_min", 0)
            print(f"{r['trial']:<10} {r['status']:<18} {r.get('iterations', '?'):>5} "
                  f"{mins:>7.1f}m {r.get('objective', 0):>12.2f} "
                  f"{r.get('max_reserve', 0):>8.2f} "
                  f"{r.get('act_mean', 0):>8.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Batch MocoInverse for BAA01 trials")
    parser.add_argument("trials", nargs="*", help="Specific trials (e.g., Walk06 Walk08)")
    parser.add_argument("--include-walk05", action="store_true",
                        help="Include Walk05 (already solved)")
    args = parser.parse_args()

    if args.trials:
        trial_names = args.trials
    elif args.include_walk05:
        trial_names = list(TRIALS.keys())
    else:
        trial_names = [t for t in TRIALS if t != "Walk05"]

    for t in trial_names:
        if t not in TRIALS:
            print(f"ERROR: Unknown trial '{t}'. Available: {list(TRIALS.keys())}")
            sys.exit(1)

    print(f"Will run {len(trial_names)} trial(s): {', '.join(trial_names)}")
    print(f"Model: {MOCO_MODEL}")
    print(f"Estimated total time: ~{len(trial_names) * 5} min")

    results = []
    total_t0 = time.time()

    for trial_name in trial_names:
        t_init, t_final = TRIALS[trial_name]
        result = run_trial(trial_name, t_init, t_final)
        results.append(result)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal batch time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hr)")
    print_summary(results)

    # Save summary
    summary_path = BASELINE_DIR / "mocoinverse_batch_summary.txt"
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_summary(results)
    sys.stdout = old_stdout
    summary_path.write_text(buffer.getvalue())
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
