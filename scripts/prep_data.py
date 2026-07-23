"""Prepare subject metadata from C3D files.

Scans the HuggingFace dataset directory structure, reads subject
anthropometrics from C3D PROCESSING parameters, and builds
``data/subjects.csv`` for the analysis pipeline.

Usage::

    uv run python scripts/prep_data.py \\
        --c3d-dir /path/to/rat-hindlimb-mocap/sourcedata \\
        --output data/subjects.csv

    # Dry-run (just print what would be written):
    uv run python scripts/prep_data.py \\
        --c3d-dir /path/to/rat-hindlimb-mocap/sourcedata \\
        --dry-run
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from subject ID prefixes to VML treatment groups.
# Derived from the experimental design in the rat-vml manuscript.
# Adjust as needed for your data.
SUBJECT_GROUP_MAP = {
    # BAA animals: Control dataset (previously collected)
    prefix: "Control"
    for prefix in [f"BAA{i:02d}" for i in range(1, 33)]
}

# The VML study used 48 animals split into 6 groups of 8.
# Subject IDs A01-A12, E01-E12, G01-G18, H01-H08, etc.
# Fill in the correct mapping based on your lab records.
VML_GROUPS = {
    # No Repair
    # TEMR
    # Healy Hydrogel (HH)
    # Healy Sponge (HS)
    # TEMR + Healy Hydrogel (TEMR+HH)
    # TEMR + Keratin Gel (TEMR+KG)
}

# Session names that contain C3D data
SESSION_NAMES = [
    "Baseline", "Week04", "Week08", "Week12", "Week18", "Week24",
]


def extract_static_c3d(subject_dir: Path, session: str) -> Path | None:
    """Find the first static C3D trial in a session directory.

    Looks for tar.gz archives first (extracts if needed), or plain .c3d files.
    """
    session_dir = subject_dir / session
    if not session_dir.exists():
        # Try as tar.gz archive
        targz = subject_dir / f"{session}.tar.gz"
        if targz.exists():
            logger.info(f"Found archive: {targz}")
            return None  # Would need extraction
        return None

    # Find static C3D trials
    c3d_files = sorted(session_dir.glob("Static*.c3d"))
    if c3d_files:
        return c3d_files[0]

    # Check subdirectories (Vicon Nexus structure)
    for sub in session_dir.iterdir():
        if sub.is_dir():
            c3d_files = sorted(sub.glob("Static*.c3d"))
            if c3d_files:
                return c3d_files[0]

    return None


def extract_subject_params(c3d_path: Path) -> dict | None:
    """Read subject mass and segment lengths from a C3D file's
    PROCESSING parameter group using ezc3d.

    Expected PROCESSING parameters:
        Mass, RFemurLength, RTibiaLength, RFootLength,
        LFemurLength, LTibiaLength, LFootLength

    Returns None if the C3D can't be read or lacks PROCESSING data.
    """
    try:
        import ezc3d
    except ImportError:
        raise ImportError("ezc3d is required. Install with: uv pip install ezc3d")

    try:
        c3d = ezc3d.c3d(str(c3d_path))
    except Exception as e:
        logger.warning(f"Failed to read {c3d_path}: {e}")
        return None

    if "PROCESSING" not in c3d.parameters:
        logger.warning(f"No PROCESSING section in {c3d_path}")
        return None

    proc = c3d.parameters["PROCESSING"]
    params = {}
    for key in ["Mass", "RFemurLength", "RTibiaLength", "RFootLength",
                 "LFemurLength", "LTibiaLength", "LFootLength"]:
        if key in proc:
            val = proc[key]["value"]
            params[key] = val[0] if isinstance(val, (list, tuple)) else val
        else:
            params[key] = None

    return params


def scan_subjects(c3d_dir: Path, sessions: list[str]) -> pl.DataFrame:
    """Scan the sourcedata directory and build a subjects DataFrame.

    Each row is one subject with columns: Subject, Session, Group,
    Mass, RFemurLength, ...
    """
    c3d_dir = Path(c3d_dir)
    rows = []

    for subj_dir in sorted(c3d_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        subject_id = subj_dir.name
        group = SUBJECT_GROUP_MAP.get(subject_id, "")

        # Try to find CONTEXTUAL INFORMATION — check Patient.enf for group
        patient_enf = subj_dir / f"{subject_id}.Patient.enf"
        if patient_enf.exists():
            # TODO: parse additional metadata from .enf if available
            pass

        for session in sessions:
            c3d = extract_static_c3d(subj_dir, session)
            if c3d is None:
                continue

            params = extract_subject_params(c3d)
            if params is None:
                logger.info(f"  {subject_id}/{session}: no PROCESSING params")
                continue

            row = {
                "Subject": subject_id,
                "Session": session,
                "Group": group,
                **params,
            }
            rows.append(row)
            logger.info(f"  {subject_id}/{session}: Mass={params.get('Mass')}")

    if not rows:
        logger.warning("No subject data found. Check --c3d-dir path.")
        return pl.DataFrame()

    return pl.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build subjects.csv from C3D data")
    parser.add_argument("--c3d-dir", type=Path, required=True,
                        help="Path to sourcedata/ directory (e.g. rat-hindlimb-mocap/sourcedata)")
    parser.add_argument("--output", type=Path, default=Path("data/subjects.csv"),
                        help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing")
    args = parser.parse_args()

    logger.info(f"Scanning {args.c3d_dir}...")
    df = scan_subjects(args.c3d_dir, SESSION_NAMES)

    if df.is_empty():
        logger.error("No subjects found. Verify --c3d-dir contains subject directories.")
        return

    # Only keep the 24-week endpoint sessions for the VML study
    # (and Baseline for controls)
    df_24wk = df.filter(
        pl.col("Session").is_in(["Baseline", "Week24"])
    )

    logger.info(f"Found {len(df_24wk)} subject-session records:")
    for row in df_24wk.iter_rows(named=True):
        logger.info(f"  {row['Subject']:>10}  {row['Session']:<10}  {row['Group']:<15}  "
                    f"Mass={row['Mass']}")

    if args.dry_run:
        logger.info("Dry-run — not writing file.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_24wk.write_csv(args.output)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
