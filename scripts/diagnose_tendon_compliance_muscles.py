import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/hudson/rat-hindlimb-model")
sys.path.insert(0, "/home/hudson/osimpy/src")

import opensim as osim

BASELINE_DIR = Path(os.getenv("CMC_BASELINE_DIR", "/home/hudson/Downloads/CMC_Runs/BAA01/Baseline"))
MOCO_MODEL = BASELINE_DIR / "scaled_moco.osim"
IK_FILE = BASELINE_DIR / "BAA01_Baseline_Walk05_ik.mot"
FP_SETUP = BASELINE_DIR / "BAA01_Baseline_Walk05_fp_setup.xml"
OUTPUT_CSV = BASELINE_DIR / "tendon_compliance_muscle_diagnostic_walk05.csv"

T_INIT = 2.87
T_FINAL = 3.45
IK_FILTER_CUTOFF_HZ = 15.0
ABS_SHORT_MARGIN_M = 0.0005
REL_SHORT_MARGIN = 0.05
SAFE_PENNATION_RAD = math.radians(75.0)


def build_model_and_table():
    model_proc = osim.ModelProcessor(str(MOCO_MODEL))
    model_proc.append(osim.ModOpAddExternalLoads(str(FP_SETUP)))
    model_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_proc.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    model_proc.append(osim.ModOpAddReserves(0.1))

    model = model_proc.process()
    state = model.initSystem()

    kinematics = osim.TableProcessor(str(IK_FILE))
    kinematics.append(osim.TabOpLowPassFilter(IK_FILTER_CUTOFF_HZ))
    kinematics.append(osim.TabOpConvertDegreesToRadians())
    kinematics.append(osim.TabOpUseAbsoluteStateNames())
    table = osim.TimeSeriesTable(kinematics.process(model))
    return model, state, table


def set_state_from_row(model, state, table, row_idx):
    time_value = table.getIndependentColumn()[row_idx]
    state.setTime(time_value)
    for label in table.getColumnLabels():
        model.setStateVariableValue(state, label, table.getDependentColumn(label)[row_idx])
    model.realizePosition(state)
    model.realizeVelocity(state)
    model.realizeDynamics(state)


def evaluate_muscle_frame(muscle, state):
    mtu_length = muscle.getLength(state)
    tendon_slack = muscle.getTendonSlackLength()
    optimal_fiber_length = muscle.getOptimalFiberLength()
    pennation_opt = muscle.getPennationAngleAtOptimalFiberLength()
    height = optimal_fiber_length * math.sin(pennation_opt)

    short_margin = mtu_length - (tendon_slack + height)
    min_short_allowed = -max(ABS_SHORT_MARGIN_M, REL_SHORT_MARGIN * optimal_fiber_length)

    safe_fiber_length = max(
        0.5 * optimal_fiber_length,
        height / math.sin(SAFE_PENNATION_RAD) if height > 0 else 0.5 * optimal_fiber_length,
    )
    fiber_along_tendon_safe = math.sqrt(max(safe_fiber_length * safe_fiber_length - height * height, 0.0))
    tendon_length_safe_max = mtu_length - fiber_along_tendon_safe
    tendon_strain_safe_max = tendon_length_safe_max / tendon_slack - 1.0 if tendon_slack > 0 else -1.0

    tendon_length = muscle.getTendonLength(state)
    tendon_strain = muscle.getTendonStrain(state)
    normalized_fiber_length = muscle.getNormalizedFiberLength(state)
    pennation = muscle.getPennationAngle(state)
    passive_force = muscle.getPassiveFiberForce(state)
    fiber_force = muscle.getFiberForce(state)
    passive_ratio = passive_force / fiber_force if abs(fiber_force) > 1e-8 else 0.0

    return {
        "short_margin": short_margin,
        "short_violation": short_margin < min_short_allowed,
        "safe_tendon_strain_max": tendon_strain_safe_max,
        "safe_capacity": max(tendon_strain_safe_max, 0.0),
        "tendon_length": tendon_length,
        "tendon_slack": tendon_slack,
        "tendon_buckling": tendon_length < tendon_slack,
        "tendon_strain": tendon_strain,
        "normalized_fiber_length": normalized_fiber_length,
        "fiber_short": normalized_fiber_length < 0.5,
        "fiber_long": normalized_fiber_length > 1.8,
        "pennation": pennation,
        "pennation_high": pennation > SAFE_PENNATION_RAD,
        "passive_ratio": passive_ratio,
    }


def summarize_metric(values, threshold):
    sorted_values = sorted(values)
    n = len(sorted_values)
    p10_idx = min(n - 1, max(0, int(0.1 * (n - 1))))
    fraction_below = sum(1 for value in sorted_values if value < threshold) / n
    return sorted_values[p10_idx], fraction_below


def classify_result(result):
    if result["geometry_violation"] or result["buckling_frames"] > 0:
        return "parameter/path review"
    if result["safe_capacity_frac_lt_0p1"] > 0.10 or result["safe_capacity_frac_lt_0p3"] > 0.25:
        return "ignore compliance candidate"
    return "low risk"


def main():
    model, state, table = build_model_and_table()
    muscles = model.getMuscles()
    times = table.getIndependentColumn()

    results = []
    for imusc in range(muscles.getSize()):
        muscle = muscles.get(imusc)
        frame_metrics = []
        for row_idx in range(table.getNumRows()):
            set_state_from_row(model, state, table, row_idx)
            frame_metrics.append(evaluate_muscle_frame(muscle, state))

        safe_capacities = [m["safe_capacity"] for m in frame_metrics]
        safe_capacity_p10, frac_lt_0p1 = summarize_metric(safe_capacities, 0.1)
        _, frac_lt_0p3 = summarize_metric(safe_capacities, 0.3)

        min_short_margin = min(m["short_margin"] for m in frame_metrics)
        result = {
            "muscle": muscle.getName(),
            "min_short_margin_mm": 1000.0 * min_short_margin,
            "safe_capacity_p10": safe_capacity_p10,
            "safe_capacity_frac_lt_0p1": frac_lt_0p1,
            "safe_capacity_frac_lt_0p3": frac_lt_0p3,
            "buckling_frames": sum(1 for m in frame_metrics if m["tendon_buckling"]),
            "high_pennation_frames": sum(1 for m in frame_metrics if m["pennation_high"]),
            "fiber_short_frames": sum(1 for m in frame_metrics if m["fiber_short"]),
            "fiber_long_frames": sum(1 for m in frame_metrics if m["fiber_long"]),
            "max_passive_ratio": max(m["passive_ratio"] for m in frame_metrics),
            "geometry_violation": any(m["short_violation"] for m in frame_metrics),
        }
        result["action"] = classify_result(result)
        results.append(result)

    results.sort(
        key=lambda row: (
            row["action"] != "parameter/path review",
            row["action"] != "ignore compliance candidate",
            row["safe_capacity_p10"],
            -row["buckling_frames"],
            row["min_short_margin_mm"],
        )
    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "muscle",
                "min_short_margin_mm",
                "safe_capacity_p10",
                "safe_capacity_frac_lt_0p1",
                "safe_capacity_frac_lt_0p3",
                "buckling_frames",
                "high_pennation_frames",
                "fiber_short_frames",
                "fiber_long_frames",
                "max_passive_ratio",
                "geometry_violation",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote diagnostic table to: {OUTPUT_CSV}")
    print("Top suspect muscles:")
    for row in results[:15]:
        print(
            f"  {row['muscle']:20s}  action={row['action']:26s}  "
            f"min_margin_mm={row['min_short_margin_mm']:+.3f}  "
            f"Fsafe_p10={row['safe_capacity_p10']:.3f}  "
            f"frac<0.1={row['safe_capacity_frac_lt_0p1']:.2%}  "
            f"frac<0.3={row['safe_capacity_frac_lt_0p3']:.2%}"
        )


if __name__ == "__main__":
    main()
