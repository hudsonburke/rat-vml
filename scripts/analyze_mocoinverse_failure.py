import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import opensim as osim


BASELINE_DIR = Path(os.getenv("CMC_BASELINE_DIR", "/home/hudson/Downloads/CMC_Runs/BAA01/Baseline"))
DEFAULT_MODEL = BASELINE_DIR / "scaled_moco.osim"
DEFAULT_FP_SETUP = BASELINE_DIR / "BAA01_Baseline_Walk05_fp_setup.xml"
DEFAULT_TSL_COMPARISON = Path(
    "/home/hudson/rat-hindlimb-model/data/parameters/tsl_comparison.csv"
)

IK_FILTER_CUTOFF_HZ = 15.0
GENERAL_RESERVE_OPTIMAL_FORCE = 0.1
SACRUM_XZ_RESERVE_OPTIMAL_FORCE = 0.1
SACRUM_Y_RESERVE_OPTIMAL_FORCE = 1.5
SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE = 0.1
GENERAL_RESERVE_WEIGHT = 10.0
SACRUM_TRANSLATION_RESERVE_WEIGHT = 0.01
SACRUM_PITCH_RESERVE_WEIGHT = 10.0
SACRUM_ROLL_RESERVE_WEIGHT = 10.0
SACRUM_YAW_RESERVE_WEIGHT = 10.0

ABS_SHORT_MARGIN_M = 0.0005
REL_SHORT_MARGIN = 0.05
SAFE_PENNATION_RAD = math.radians(75.0)

SACRUM_X_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_x"
SACRUM_Y_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_y"
SACRUM_Z_RESERVE_NAME = "reserve_jointset_ground_spine_sacrum_z"
SACRUM_ROTATION_RESERVE_NAMES = {
    "reserve_jointset_ground_spine_sacrum_pitch",
    "reserve_jointset_ground_spine_sacrum_roll",
    "reserve_jointset_ground_spine_sacrum_yaw",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze a failed or questionable compliant MocoInverse solution."
    )
    parser.add_argument("--solution", required=True, help="Path to MocoInverse .sto solution")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Base OpenSim model path")
    parser.add_argument(
        "--fp-setup",
        default=str(DEFAULT_FP_SETUP),
        help="External loads XML used for the inverse solve",
    )
    parser.add_argument(
        "--tsl-comparison",
        default=str(DEFAULT_TSL_COMPARISON),
        help="TSL comparison CSV used to choose rigid-tendon subset",
    )
    parser.add_argument(
        "--tsl-ignore-threshold-mm",
        type=float,
        default=0.5,
        help="Walk TSL threshold used to ignore tendon compliance",
    )
    parser.add_argument(
        "--fiber-damping",
        type=float,
        default=0.01,
        help="Runtime DGF fiber damping used to rebuild the analysis model",
    )
    parser.add_argument(
        "--active-force-width-scale",
        type=float,
        default=1.5,
        help="Runtime DGF active force-length width scale used to rebuild the analysis model",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top suspect muscles to print in the summary",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output reports (defaults to solution parent)",
    )
    parser.add_argument(
        "--diagnostic-muscle",
        default=None,
        help="Optional muscle name for per-frame tendon/slack diagnostics (e.g. R_OI)",
    )
    return parser.parse_args(argv)


def load_thresholded_ignore_tendon_compliance_muscles(tsl_csv_path, threshold_mm):
    tsl_csv_path = Path(tsl_csv_path)
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


def parse_storage_file(file_path):
    file_path = Path(file_path)
    with open(file_path) as f:
        lines = f.readlines()

    header = {}
    header_end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "endheader":
            header_end = idx
            break
        if "=" in stripped and not stripped.startswith("time"):
            key, value = stripped.split("=", 1)
            header[key.strip()] = value.strip()

    if header_end is None or header_end + 1 >= len(lines):
        raise ValueError(f"Malformed storage file: {file_path}")

    column_names = lines[header_end + 1].strip().split("\t")
    data_rows = []
    for line in lines[header_end + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split("\t")
        if len(values) != len(column_names):
            continue
        data_rows.append([float(value) for value in values])

    data = np.array(data_rows, dtype=float)
    if data.size == 0:
        raise ValueError(f"No tabular data found in {file_path}")
    return header, column_names, data


def get_reserve_optimal_force(reserve_name):
    short_name = reserve_name.split("/")[-1]
    if short_name == SACRUM_Y_RESERVE_NAME:
        return SACRUM_Y_RESERVE_OPTIMAL_FORCE
    if short_name in {SACRUM_X_RESERVE_NAME, SACRUM_Z_RESERVE_NAME}:
        return SACRUM_XZ_RESERVE_OPTIMAL_FORCE
    if short_name in SACRUM_ROTATION_RESERVE_NAMES:
        return SACRUM_ROTATION_RESERVE_OPTIMAL_FORCE
    return GENERAL_RESERVE_OPTIMAL_FORCE


def get_reserve_weight(reserve_name):
    short_name = reserve_name.split("/")[-1]
    if short_name in {SACRUM_X_RESERVE_NAME, SACRUM_Y_RESERVE_NAME, SACRUM_Z_RESERVE_NAME}:
        return SACRUM_TRANSLATION_RESERVE_WEIGHT
    if short_name.endswith("sacrum_pitch"):
        return SACRUM_PITCH_RESERVE_WEIGHT
    if short_name.endswith("sacrum_roll"):
        return SACRUM_ROLL_RESERVE_WEIGHT
    if short_name.endswith("sacrum_yaw"):
        return SACRUM_YAW_RESERVE_WEIGHT
    return GENERAL_RESERVE_WEIGHT


def percentile(values, p):
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def time_at_extreme(times, values, use_abs=False, mode="max"):
    arr = np.abs(values) if use_abs else np.asarray(values, dtype=float)
    index = int(np.argmax(arr) if mode == "max" else np.argmin(arr))
    return float(times[index])


def build_runtime_model(args):
    ignore_tendon_compliance = load_thresholded_ignore_tendon_compliance_muscles(
        args.tsl_comparison, args.tsl_ignore_threshold_mm
    )

    model_proc = osim.ModelProcessor(str(args.model))
    model_proc.append(osim.ModOpAddExternalLoads(str(args.fp_setup)))
    model_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_proc.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(args.active_force_width_scale))
    model_proc.append(osim.ModOpAddReserves(GENERAL_RESERVE_OPTIMAL_FORCE))

    model = model_proc.process()
    model.initSystem()

    muscles = model.updMuscles()
    for index in range(muscles.getSize()):
        muscle = muscles.get(index)
        dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
        if dgf is None:
            continue
        dgf.set_fiber_damping(args.fiber_damping)
        if muscle.getName() in ignore_tendon_compliance:
            dgf.set_ignore_tendon_compliance(True)

    forces = model.updForceSet()
    for index in range(forces.getSize()):
        actuator = osim.CoordinateActuator.safeDownCast(forces.get(index))
        if actuator is None:
            continue
        name = actuator.getName()
        actuator.setOptimalForce(get_reserve_optimal_force(name))

    state = model.initSystem()
    return model, state, ignore_tendon_compliance


def summarize_reserves(times, column_names, data):
    reserve_rows = []
    for index, column_name in enumerate(column_names):
        if index == 0 or "reserve" not in column_name.lower():
            continue
        values = data[:, index]
        short_name = column_name.split("/")[-1]
        reserve_rows.append(
            {
                "reserve_name": column_name,
                "coordinate": short_name.replace("reserve_jointset_", ""),
                "max_abs_control": float(np.max(np.abs(values))),
                "p95_abs_control": percentile(np.abs(values), 95),
                "mean_abs_control": float(np.mean(np.abs(values))),
                "time_at_max_abs_control": time_at_extreme(times, values, use_abs=True),
                "reserve_optimal_force_setting": get_reserve_optimal_force(short_name),
                "reserve_weight_setting": get_reserve_weight(short_name),
                "max_abs_generalized_force": float(
                    np.max(np.abs(values)) * get_reserve_optimal_force(short_name)
                ),
                "p95_abs_generalized_force": float(
                    percentile(np.abs(values), 95) * get_reserve_optimal_force(short_name)
                ),
                "mean_abs_generalized_force": float(
                    np.mean(np.abs(values)) * get_reserve_optimal_force(short_name)
                ),
            }
        )

    reserve_rows.sort(key=lambda row: row["max_abs_generalized_force"], reverse=True)
    return reserve_rows


def build_muscle_signal_summary(model, column_names, data, times, rigid_muscles):
    muscles = model.getMuscles()
    summary = {}
    for index in range(muscles.getSize()):
        muscle = muscles.get(index)
        path = muscle.getAbsolutePathString()
        activation_label = path + "/activation"
        ntf_label = path + "/normalized_tendon_force"

        activation_values = None
        ntf_values = None
        if activation_label in column_names:
            activation_values = data[:, column_names.index(activation_label)]
        if ntf_label in column_names:
            ntf_values = data[:, column_names.index(ntf_label)]

        row = {
            "muscle": muscle.getName(),
            "max_isometric_force": float(muscle.getMaxIsometricForce()),
            "ignore_tendon_compliance": muscle.getName() in rigid_muscles,
            "has_normalized_tendon_force": ntf_values is not None,
        }

        if activation_values is not None:
            row.update(
                {
                    "max_activation": float(np.max(activation_values)),
                    "p95_activation": percentile(activation_values, 95),
                    "frac_activation_gt_0p90": float(np.mean(activation_values > 0.90)),
                    "frac_activation_gt_0p95": float(np.mean(activation_values > 0.95)),
                    "time_at_max_activation": time_at_extreme(times, activation_values),
                }
            )
        else:
            row.update(
                {
                    "max_activation": float("nan"),
                    "p95_activation": float("nan"),
                    "frac_activation_gt_0p90": float("nan"),
                    "frac_activation_gt_0p95": float("nan"),
                    "time_at_max_activation": float("nan"),
                }
            )

        if ntf_values is not None:
            row.update(
                {
                    "max_normalized_tendon_force": float(np.max(ntf_values)),
                    "p95_normalized_tendon_force": percentile(ntf_values, 95),
                    "frac_ntf_gt_1p0": float(np.mean(ntf_values > 1.0)),
                    "frac_ntf_gt_1p2": float(np.mean(ntf_values > 1.2)),
                    "time_at_max_ntf": time_at_extreme(times, ntf_values),
                    "max_demand_force_N": float(np.max(ntf_values) * muscle.getMaxIsometricForce()),
                }
            )
        else:
            row.update(
                {
                    "max_normalized_tendon_force": float("nan"),
                    "p95_normalized_tendon_force": float("nan"),
                    "frac_ntf_gt_1p0": float("nan"),
                    "frac_ntf_gt_1p2": float("nan"),
                    "time_at_max_ntf": float("nan"),
                    "max_demand_force_N": float("nan"),
                }
            )

        summary[muscle.getName()] = row
    return summary


def safe_ratio(numerator, denominator):
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def sanitize_filename_component(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


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
    fiber_force = muscle.getFiberForce(state)
    max_iso = muscle.getMaxIsometricForce()

    return {
        "short_margin": short_margin,
        "short_violation": short_margin < min_short_allowed,
        "safe_capacity": max(tendon_strain_safe_max, 0.0),
        "tendon_length": tendon_length,
        "tendon_slack": tendon_slack,
        "tendon_length_minus_slack": tendon_length - tendon_slack,
        "tendon_length_over_slack": safe_ratio(tendon_length, tendon_slack),
        "tendon_buckling": tendon_length < tendon_slack,
        "tendon_strain": tendon_strain,
        "normalized_fiber_length": normalized_fiber_length,
        "fiber_short": normalized_fiber_length < 0.5,
        "fiber_long": normalized_fiber_length > 1.8,
        "pennation": pennation,
        "pennation_high": pennation > SAFE_PENNATION_RAD,
        "fiber_force_over_max_iso": abs(fiber_force) / max_iso if max_iso > 0 else 0.0,
    }


def get_model_state_names(model):
    state_names = model.getStateVariableNames()
    return {state_names.get(index) for index in range(state_names.getSize())}


def build_selected_muscle_frame_rows(muscle_name, frames):
    rows = []
    for frame in frames:
        rows.append(
            {
                "muscle": muscle_name,
                "time": float(frame["time"]),
                "tendon_length_m": float(frame["tendon_length"]),
                "tendon_slack_m": float(frame["tendon_slack"]),
                "tendon_length_minus_slack_m": float(frame["tendon_length_minus_slack"]),
                "tendon_length_minus_slack_mm": 1000.0 * float(frame["tendon_length_minus_slack"]),
                "tendon_length_over_slack": float(frame["tendon_length_over_slack"]),
                "tendon_buckling": bool(frame["tendon_buckling"]),
                "tendon_strain": float(frame["tendon_strain"]),
                "safe_capacity": float(frame["safe_capacity"]),
                "short_margin_m": float(frame["short_margin"]),
                "short_margin_mm": 1000.0 * float(frame["short_margin"]),
                "normalized_fiber_length": float(frame["normalized_fiber_length"]),
                "pennation_deg": float(math.degrees(frame["pennation"])),
                "fiber_force_over_max_iso": float(frame["fiber_force_over_max_iso"]),
            }
        )
    return rows


def build_geometry_summary(model, state, column_names, data, times, diagnostic_muscle=None):
    allowed_state_names = get_model_state_names(model)
    state_indices = [
        index for index, name in enumerate(column_names) if name in allowed_state_names
    ]
    state_names = [column_names[index] for index in state_indices]

    geometry_frames = {model.getMuscles().get(i).getName(): [] for i in range(model.getMuscles().getSize())}

    for row_index in range(data.shape[0]):
        state.setTime(float(times[row_index]))
        for index, name in zip(state_indices, state_names):
            model.setStateVariableValue(state, name, float(data[row_index, index]))
        model.realizePosition(state)
        model.realizeVelocity(state)
        model.realizeDynamics(state)

        muscles = model.getMuscles()
        for muscle_index in range(muscles.getSize()):
            muscle = muscles.get(muscle_index)
            metrics = evaluate_muscle_frame(muscle, state)
            metrics["time"] = float(times[row_index])
            geometry_frames[muscle.getName()].append(metrics)

    summary = {}
    for muscle_name, frames in geometry_frames.items():
        safe_capacities = [frame["safe_capacity"] for frame in frames]
        short_margins = [frame["short_margin"] for frame in frames]
        tendon_strains = [frame["tendon_strain"] for frame in frames]
        normalized_fiber_lengths = [frame["normalized_fiber_length"] for frame in frames]
        pennations = [frame["pennation"] for frame in frames]
        force_over_max = [frame["fiber_force_over_max_iso"] for frame in frames]

        min_short_index = int(np.argmin(short_margins))
        summary[muscle_name] = {
            "min_short_margin_mm": 1000.0 * float(min(short_margins)),
            "geometry_violation": any(frame["short_violation"] for frame in frames),
            "safe_capacity_p10": percentile(safe_capacities, 10),
            "safe_capacity_frac_lt_0p1": float(np.mean(np.asarray(safe_capacities) < 0.1)),
            "safe_capacity_frac_lt_0p3": float(np.mean(np.asarray(safe_capacities) < 0.3)),
            "buckling_frames": int(sum(frame["tendon_buckling"] for frame in frames)),
            "high_pennation_frames": int(sum(frame["pennation_high"] for frame in frames)),
            "fiber_short_frames": int(sum(frame["fiber_short"] for frame in frames)),
            "fiber_long_frames": int(sum(frame["fiber_long"] for frame in frames)),
            "max_tendon_strain": float(max(tendon_strains)),
            "max_normalized_fiber_length": float(max(normalized_fiber_lengths)),
            "min_normalized_fiber_length": float(min(normalized_fiber_lengths)),
            "max_pennation_deg": float(math.degrees(max(pennations))),
            "max_fiber_force_over_max_iso": float(max(force_over_max)),
            "time_at_min_short_margin": float(frames[min_short_index]["time"]),
        }

    diagnostic_rows = []
    if diagnostic_muscle is not None:
        diagnostic_rows = build_selected_muscle_frame_rows(
            diagnostic_muscle, geometry_frames[diagnostic_muscle]
        )

    return summary, diagnostic_rows


def classify_muscle_issue(row):
    hard_geometry_signal = (
        bool(row["geometry_violation"])
        or row["buckling_frames"] > 0
        or row["high_pennation_frames"] > 0
        or row["fiber_short_frames"] > 0
        or row["fiber_long_frames"] > 0
    )
    soft_geometry_warning = row["safe_capacity_frac_lt_0p1"] > 0.10 or row["safe_capacity_frac_lt_0p3"] > 0.25

    activation_signal = (
        row["max_activation"] == row["max_activation"]
        and (row["frac_activation_gt_0p95"] > 0.10 or row["max_activation"] > 0.90)
    )
    ntf_signal = (
        row["has_normalized_tendon_force"]
        and (
            row["p95_normalized_tendon_force"] > 0.90
            or row["max_normalized_tendon_force"] > 1.2
        )
    )
    capacity_signal = activation_signal and (ntf_signal or not row["has_normalized_tendon_force"])

    reasons = []
    if hard_geometry_signal:
        if row["geometry_violation"]:
            reasons.append("short-margin violation")
        if row["buckling_frames"] > 0:
            reasons.append("tendon buckling")
        if row["high_pennation_frames"] > 0:
            reasons.append("high pennation")
        if row["fiber_short_frames"] > 0:
            reasons.append("fiber too short")
        if row["fiber_long_frames"] > 0:
            reasons.append("fiber too long")
    elif soft_geometry_warning:
        reasons.append("low geometry headroom warning")
    if capacity_signal:
        if row["max_activation"] > 0.90:
            reasons.append("activation saturation")
        if row["has_normalized_tendon_force"] and row["max_normalized_tendon_force"] > 1.2:
            reasons.append("high normalized tendon force")
        elif not row["has_normalized_tendon_force"]:
            reasons.append("high activation on rigid tendon muscle")

    if hard_geometry_signal and capacity_signal:
        label = "mixed"
    elif hard_geometry_signal:
        label = "geometry_path"
    elif capacity_signal:
        label = "force_capacity"
    else:
        label = "low_signal"

    score = 0.0
    score += 5.0 if row["geometry_violation"] else 0.0
    score += 0.5 * row["safe_capacity_frac_lt_0p1"]
    score += 0.25 * row["safe_capacity_frac_lt_0p3"]
    score += 1.0 * row["buckling_frames"]
    score += 1.0 * row["high_pennation_frames"]
    score += 2.0 * max(0.0, row["max_activation"] - 0.9)
    if row["has_normalized_tendon_force"] and row["max_normalized_tendon_force"] == row["max_normalized_tendon_force"]:
        score += 2.0 * max(0.0, row["max_normalized_tendon_force"] - 1.0)

    row["issue_label"] = label
    row["why_flagged"] = "; ".join(reasons) if reasons else "no strong signal"
    row["suspicion_score"] = float(score)
    return row


def merge_muscle_summaries(signal_summary, geometry_summary):
    rows = []
    for muscle_name, signal_row in signal_summary.items():
        merged = dict(signal_row)
        merged.update(geometry_summary[muscle_name])
        rows.append(classify_muscle_issue(merged))
    rows.sort(
        key=lambda row: (
            row["issue_label"] == "low_signal",
            -row["suspicion_score"],
            -row["max_activation"],
        )
    )
    return rows


def classify_run_issue(muscle_rows, reserve_rows):
    counts = {"force_capacity": 0, "geometry_path": 0, "mixed": 0, "low_signal": 0}
    for row in muscle_rows:
        counts[row["issue_label"]] += 1

    top_reserve = reserve_rows[0]["max_abs_generalized_force"] if reserve_rows else 0.0
    high_reserves = sum(row["max_abs_generalized_force"] > 0.10 for row in reserve_rows)

    if counts["geometry_path"] + counts["mixed"] >= counts["force_capacity"] + 6 and high_reserves <= 2:
        verdict = "likely geometry/path limited"
    elif counts["force_capacity"] + counts["mixed"] >= counts["geometry_path"] + 4:
        verdict = "likely force-capacity limited"
    elif high_reserves > 0 and top_reserve > 0.10:
        verdict = "likely formulation/coordinate issue"
    else:
        verdict = "mixed/unclear"

    return verdict, counts


def write_csv(file_path, fieldnames, rows):
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)


def build_summary_text(solution_path, header, verdict, issue_counts, reserve_rows, muscle_rows, top_n):
    lines = []
    lines.append(f"Failure analysis for: {solution_path}")
    lines.append(f"Run verdict: {verdict}")
    lines.append("")
    lines.append("Header summary:")
    for key in [
        "status",
        "success",
        "objective",
        "objective_excitation_effort",
        "objective_accelerations",
        "objective_auxiliary_derivatives",
        "num_iterations",
        "solver_duration",
        "num_states",
        "num_controls",
        "num_derivatives",
    ]:
        if key in header:
            lines.append(f"  {key}: {header[key]}")
    lines.append("")
    lines.append(
        "Issue counts: "
        + ", ".join(f"{key}={value}" for key, value in issue_counts.items())
    )
    lines.append("")
    lines.append("Top reserve coordinates:")
    for row in reserve_rows[:10]:
        lines.append(
            "  "
            f"{row['reserve_name']}  max|u|={row['max_abs_control']:.4f}  "
            f"max|tau|={row['max_abs_generalized_force']:.4f}  "
            f"p95|tau|={row['p95_abs_generalized_force']:.4f}  "
            f"optimal_force={row['reserve_optimal_force_setting']:.3f}"
        )
    lines.append("")
    lines.append(f"Top suspect muscles (top {top_n}):")
    for row in muscle_rows[:top_n]:
        lines.append(
            "  "
            f"{row['muscle']:20s}  issue={row['issue_label']:14s}  "
            f"score={row['suspicion_score']:.3f}  "
            f"act_max={format_float(row['max_activation'])}  "
            f"ntf_max={format_float(row['max_normalized_tendon_force'])}  "
            f"min_margin_mm={format_float(row['min_short_margin_mm'])}  "
            f"why={row['why_flagged']}"
        )
    lines.append("")
    lines.append(
        "Note: muscles in the rigid-tendon subset do not carry a normalized_tendon_force state."
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    solution_path = Path(args.solution)
    output_dir = Path(args.output_dir) if args.output_dir else solution_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    header, column_names, data = parse_storage_file(solution_path)
    times = data[:, 0]

    model, state, rigid_muscles = build_runtime_model(args)
    if args.diagnostic_muscle is not None:
        muscle_names = {
            model.getMuscles().get(index).getName()
            for index in range(model.getMuscles().getSize())
        }
        if args.diagnostic_muscle not in muscle_names:
            raise ValueError(
                f"Diagnostic muscle {args.diagnostic_muscle!r} not found in model. "
                f"Available example names: {', '.join(sorted(list(muscle_names))[:10])}"
            )

    reserve_rows = summarize_reserves(times, column_names, data)
    signal_summary = build_muscle_signal_summary(model, column_names, data, times, rigid_muscles)
    geometry_summary, diagnostic_rows = build_geometry_summary(
        model,
        state,
        column_names,
        data,
        times,
        diagnostic_muscle=args.diagnostic_muscle,
    )
    muscle_rows = merge_muscle_summaries(signal_summary, geometry_summary)

    verdict, issue_counts = classify_run_issue(muscle_rows, reserve_rows)

    stem = solution_path.stem
    reserve_path = output_dir / f"{stem}_reserve_diagnostics.csv"
    muscle_path = output_dir / f"{stem}_muscle_diagnostics.csv"
    summary_path = output_dir / f"{stem}_failure_summary.txt"
    diagnostic_path = None
    if diagnostic_rows:
        suffix = sanitize_filename_component(args.diagnostic_muscle)
        diagnostic_path = output_dir / f"{stem}_{suffix}_frame_diagnostics.csv"

    if reserve_rows:
        write_csv(reserve_path, list(reserve_rows[0].keys()), reserve_rows)
    else:
        reserve_path.write_text("reserve_name,coordinate,max_abs_control\n")

    write_csv(muscle_path, list(muscle_rows[0].keys()), muscle_rows)
    if diagnostic_rows and diagnostic_path is not None:
        write_csv(diagnostic_path, list(diagnostic_rows[0].keys()), diagnostic_rows)

    summary_text = build_summary_text(
        solution_path=solution_path,
        header=header,
        verdict=verdict,
        issue_counts=issue_counts,
        reserve_rows=reserve_rows,
        muscle_rows=muscle_rows,
        top_n=args.top_n,
    )
    summary_path.write_text(summary_text)
    print(summary_text, end="")
    print(f"Wrote reserve diagnostics to: {reserve_path}")
    print(f"Wrote muscle diagnostics to: {muscle_path}")
    if diagnostic_rows and diagnostic_path is not None:
        print(f"Wrote selected muscle frame diagnostics to: {diagnostic_path}")
    print(f"Wrote text summary to: {summary_path}")


if __name__ == "__main__":
    main()
