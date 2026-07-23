"""MoveDB catalog integration for the rat-vml pipeline.

Uses movedb-core to import C3D data into Rerun .rrd files, then queries
the catalog with DuckDB to find valid walking trials and build
subjects.csv with treatment group tags.

Usage::

    # 1. Import all C3D data to .rrd (one .rrd per subject)
    uv run python scripts/catalog.py import /path/to/sourcedata -o data/rrd/

    # 2. Query to find valid walking trials
    uv run python scripts/catalog.py query data/rrd/ --side right

    # 3. Build subjects.csv from the catalog
    uv run python scripts/catalog.py subjects data/rrd/ -o data/subjects.csv

    # 4. Start Rerun catalog server for visualization
    uv run python scripts/catalog.py serve data/rrd/
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Treatment group mapping from AFIRM spreadsheet
try:
    from rathindlimb.analysis.subject_groups import SUBJECT_TO_GROUP
except ImportError:
    SUBJECT_TO_GROUP = {}


def run_import(c3d_dir: str, output_dir: str, min_body_measurements: int = 3, group_map_path: str | None = None) -> dict[str, str]:
    """Import C3D files to .rrd using movedb-core's C3D importer.

    Parameters
    ----------
    c3d_dir : str
        Root directory containing C3D files (e.g. sourcedata/).
    output_dir : str
        Output directory for .rrd files.
    min_body_measurements : int
        Minimum body measurement params before stopping scan.
    group_map_path : str or None
        Path to JSON file mapping subject IDs to treatment groups.
        If not provided, auto-generates from rathindlimb.analysis.subject_groups.

    Returns
    -------
    dict mapping subject_name → rrd_filepath
    """
    from rerun_importer_c3d.batch import batch_import, load_group_map

    # Auto-generate group map from subject_groups if not provided
    group_map = load_group_map(group_map_path)
    if not group_map and SUBJECT_TO_GROUP:
        group_map = SUBJECT_TO_GROUP
        logger.info(f"Auto-generated group map with {len(group_map)} subjects")

    logger.info(f"Importing C3D files from {c3d_dir} → {output_dir}")
    results = batch_import(c3d_dir, output_dir, min_body_measurements, group_map=group_map)
    logger.info(f"Imported {len(results)} subjects")
    return results


def query_valid_trials(
    rrd_dir: str,
    side: str = "right",
    min_events: int = 7,
) -> list[dict]:
    """Query the .rrd catalog for valid walking trials.

    A trial is valid if it has the expected 7 gait events in the correct
    alternating order on the primary side.

    Parameters
    ----------
    rrd_dir : str
        Directory containing .rrd files.
    side : str
        Primary analysis side ("left" or "right").
    min_events : int
        Minimum events required.

    Returns
    -------
    list of dicts with subject, trial, event counts, validity
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is required. Install: uv pip install duckdb")

    rrd_dir = os.path.abspath(rrd_dir)
    conn = duckdb.connect()

    # Load the Rerun DuckDB extension
    try:
        conn.execute("INSTALL rrd FROM community;")
        conn.execute("LOAD rrd;")
    except Exception:
        # Try loading from local extension
        conn.execute("LOAD rrd;")

    # Register the .rrd directory
    conn.execute(f"CALL rrd_scan_directory('{rrd_dir}', 'biomechanics');")

    # Query for events
    try:
        result = conn.execute("""
            SELECT entity_path, data
            FROM biomechanics
            WHERE entity_path LIKE '%/events/%'
            ORDER BY entity_path
        """)
        rows = result.fetchall()
        logger.info(f"Found {len(rows)} event entries in catalog")
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        logger.info("Falling back to file-based event extraction")
        return _query_events_from_files(rrd_dir, side, min_events)

    # Parse events and validate
    valid_trials = []
    for entity_path, data in rows:
        parts = entity_path.split("/")
        if len(parts) >= 3:
            subject = parts[1]
            trial = parts[2]
            # Parse event from Rerun TextLog
            # Format: "context — label @ time"
            valid_trials.append({
                "subject": subject,
                "trial": trial,
                "entity_path": entity_path,
            })

    return valid_trials


def _query_events_from_files(
    rrd_dir: str,
    side: str = "right",
    min_events: int = 7,
) -> list[dict]:
    """Fallback: extract events directly from C3D files."""
    from rathindlimb.analysis.events import extract_events_from_c3d, validate_walking_trial

    rrd_path = Path(rrd_dir)
    # Find the original C3D files (assume they're in a sibling directory)
    c3d_dir = rrd_path.parent / "raw"
    if not c3d_dir.exists():
        c3d_dir = rrd_path.parent / "sourcedata"

    results = []
    for c3d in sorted(c3d_dir.rglob("*.c3d")):
        if "static" in c3d.name.lower():
            continue

        try:
            events = extract_events_from_c3d(c3d)
            is_valid, reason = validate_walking_trial(events, side, min_events)

            # Extract subject from path (e.g. sourcedata/BAA01/Baseline/Walk01.c3d)
            parts = c3d.parts
            subject = next((p for p in parts if p.startswith(("BAA", "A0", "E0", "G0", "H0", "K0", "S0", "T0", "LGS"))), c3d.stem)

            results.append({
                "subject": subject,
                "trial": c3d.stem,
                "path": str(c3d),
                "is_valid": is_valid,
                "reason": reason,
                "n_events": len(events.right_foot_strike) + len(events.right_foot_off),
            })
        except Exception as e:
            logger.warning(f"Failed to extract events from {c3d}: {e}")

    valid = [r for r in results if r["is_valid"]]
    logger.info(f"Found {len(valid)}/{len(results)} valid trials")
    return results


def build_subjects_csv(
    rrd_dir: str,
    output_path: str,
    session: str = "Baseline",
) -> Path:
    """Build subjects.csv from the .rrd catalog.

    Extracts body measurements from each subject's .rrd file and
    tags them with treatment groups from the AFIRM spreadsheet.

    Parameters
    ----------
    rrd_dir : str
        Directory containing .rrd files (one per subject).
    output_path : str
        Output CSV path.
    session : str
        Session name to include in the CSV.

    Returns
    -------
    Path to the written CSV.
    """
    import polars as pl

    rrd_path = Path(rrd_dir)
    rows = []

    # For each .rrd file, extract body measurements
    # The .rrd files contain static scalars for body measurements
    # We'll also look for the original C3D files to get PROCESSING params
    c3d_dir = rrd_path.parent / "sourcedata"
    if not c3d_dir.exists():
        c3d_dir = rrd_path.parent / "raw"

    for rrd_file in sorted(rrd_path.glob("*.rrd")):
        subject_id = rrd_file.stem
        group = SUBJECT_TO_GROUP.get(subject_id, "")

        # Find the corresponding C3D static trial
        static_c3d = None
        subject_c3d_dir = c3d_dir / subject_id
        if subject_c3d_dir.exists():
            static_files = sorted(subject_c3d_dir.glob("*Static*.c3d"))
            if static_files:
                static_c3d = static_files[0]
            else:
                # Look in session subdirectories
                for session_dir in subject_c3d_dir.iterdir():
                    if session_dir.is_dir():
                        static_files = sorted(session_dir.glob("*Static*.c3d"))
                        if static_files:
                            static_c3d = static_files[0]
                            break

        if static_c3d is None:
            logger.warning(f"No static C3D found for {subject_id}")
            continue

        # Extract body measurements from C3D PROCESSING parameters
        try:
            import ezc3d
            c3d = ezc3d.c3d(str(static_c3d))
            proc = c3d["parameters"].get("PROCESSING", {})

            params = {"Subject": subject_id, "Session": session, "Group": group}
            for key in ["Mass", "RFemurLength", "RTibiaLength", "RFootLength",
                         "LFemurLength", "LTibiaLength", "LFootLength"]:
                if key in proc:
                    val = proc[key]["value"]
                    params[key] = val[0] if isinstance(val, (list, np.ndarray)) else val

            rows.append(params)
            logger.info(f"  {subject_id}: Mass={params.get('Mass')}, Group={group}")
        except Exception as e:
            logger.warning(f"Failed to extract params from {static_c3d}: {e}")

    if not rows:
        logger.error("No subject data found.")
        return Path(output_path)

    df = pl.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output)
    logger.info(f"Wrote {output} ({len(df)} subjects)")
    return output


def serve_catalog(rrd_dir: str, port: int = 51234, host: str = "0.0.0.0") -> None:
    """Start a Rerun catalog server for the .rrd files."""
    try:
        import rerun as rr
    except ImportError:
        raise ImportError("rerun-sdk required. Install: uv pip install rerun-sdk")

    rrd_dir = os.path.abspath(rrd_dir)
    logger.info(f"Starting catalog server: {host}:{port}")
    logger.info(f"  Dataset 'biomechanics' → {rrd_dir}")
    logger.info(f"  Connect via: rerun --connect rerun+http://{host}:{port}/proxy")

    server = rr.server.Server(
        host=host,
        port=port,
        datasets={"biomechanics": rrd_dir},
    )
    try:
        server.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


def main():
    parser = argparse.ArgumentParser(
        description="MoveDB catalog integration for rat-vml pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Import C3D data to .rrd\n"
            "  python scripts/catalog.py import /data/sourcedata -o data/rrd/\n"
            "\n"
            "  # Build subjects.csv from catalog\n"
            "  python scripts/catalog.py subjects data/rrd/ -o data/subjects.csv\n"
            "\n"
            "  # Query for valid walking trials\n"
            "  python scripts/catalog.py query data/rrd/ --side right\n"
            "\n"
            "  # Start Rerun catalog server\n"
            "  python scripts/catalog.py serve data/rrd/\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # import
    import_parser = subparsers.add_parser("import", help="Import C3D files to .rrd")
    import_parser.add_argument("c3d_dir", help="Root directory with C3D files")
    import_parser.add_argument("-o", "--output-dir", default="data/rrd", help="Output .rrd directory")
    import_parser.add_argument("--min-body-measurements", type=int, default=3)
    import_parser.add_argument("--group-map", default=None, help="JSON file mapping subject IDs to treatment groups (auto-generated from AFIRM spreadsheet if not provided)")

    # subjects
    subjects_parser = subparsers.add_parser("subjects", help="Build subjects.csv from catalog")
    subjects_parser.add_argument("rrd_dir", help="Directory with .rrd files")
    subjects_parser.add_argument("-o", "--output", default="data/subjects.csv")
    subjects_parser.add_argument("--session", default="Baseline")

    # query
    query_parser = subparsers.add_parser("query", help="Query for valid walking trials")
    query_parser.add_argument("rrd_dir", help="Directory with .rrd files")
    query_parser.add_argument("--side", default="right", choices=["left", "right"])
    query_parser.add_argument("--min-events", type=int, default=7)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start Rerun catalog server")
    serve_parser.add_argument("rrd_dir", help="Directory with .rrd files")
    serve_parser.add_argument("--port", type=int, default=51234)
    serve_parser.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "import":
        run_import(args.c3d_dir, args.output_dir, args.min_body_measurements, args.group_map)
    elif args.command == "subjects":
        build_subjects_csv(args.rrd_dir, args.output, args.session)
    elif args.command == "query":
        results = query_valid_trials(args.rrd_dir, args.side, args.min_events)
        for r in results:
            status = "✓" if r.get("is_valid") else f"✗ {r.get('reason', '')}"
            print(f"  {r['subject']:>10}  {r['trial']:<15}  {status}")
    elif args.command == "serve":
        serve_catalog(args.rrd_dir, args.port, args.host)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
