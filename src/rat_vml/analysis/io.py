"""Rerun .rrd → OpenSim file format extraction.

Extracts marker and force plate data from Rerun .rrd recordings and writes
TRC (markers) and MOT (force plates) files for OpenSim IK/ID.

This module reads from the MoveDB catalog (.rrd files) — not from C3D files.
The C3D→.rrd conversion happens once during ingestion via rerun-importer-c3d.
This module only runs after valid trials have been selected from the catalog.

Entity path convention (MoveDB standard):
    {subject}/trials/{session}_{trial}/markers              → rr.Points3D
    {subject}/trials/{session}_{trial}/force_plates/{fp}/force  → rr.Arrows3D
    {subject}/trials/{session}_{trial}/force_plates/{fp}/cop    → rr.Points3D
    {subject}/trials/{session}_{trial}/force_plates/{fp}/moment → rr.Arrows3D
    {subject}/trials/{session}_{trial}/events/{label}           → rr.TextLog
    {subject}/subject/body_measurements/{param}                 → rr.Scalars

Usage::

    from rat_vml.analysis.io import rrd_to_trc, rrd_to_fp_mot

    # Extract markers from .rrd and write TRC
    trc_path = rrd_to_trc("data/rrd/BAA01.rrd", entity_prefix, output_dir)

    # Extract force plates from .rrd and write MOT + ext loads XML
    mot_path, ext_loads_path = rrd_to_fp_mot(
        "data/rrd/BAA01.rrd", entity_prefix, events, output_dir
    )
"""

import logging
from pathlib import Path

import numpy as np

from .events import GaitEvents, VICON_TO_OPENSIM

logger = logging.getLogger(__name__)


def _get_duckdb_connection(rrd_path: str):
    """Create a DuckDB connection with the rrd extension loaded for a single .rrd file."""
    import duckdb

    conn = duckdb.connect()
    try:
        conn.execute("INSTALL rrd FROM community;")
        conn.execute("LOAD rrd;")
    except Exception:
        conn.execute("LOAD rrd;")

    conn.execute(f"CALL rrd_scan('{rrd_path}', 'recording');")
    return conn


def _get_catalog_connection(rrd_dir: str):
    """Create a DuckDB connection with the rrd extension loaded for a directory."""
    import duckdb

    conn = duckdb.connect()
    try:
        conn.execute("INSTALL rrd FROM community;")
        conn.execute("LOAD rrd;")
    except Exception:
        conn.execute("LOAD rrd;")

    conn.execute(f"CALL rrd_scan_directory('{rrd_dir}', 'recording');")
    return conn


# -----------------------------------------------------------------------
# Marker extraction from .rrd
# -----------------------------------------------------------------------

def extract_markers_from_rrd(
    rrd_path: str | Path,
    entity_prefix: str,
    time_range: tuple[float, float] | None = None,
) -> tuple[list[str], np.ndarray, float]:
    """Extract marker positions from a Rerun .rrd recording.

    Parameters
    ----------
    rrd_path : str or Path
        Path to the .rrd file.
    entity_prefix : str
        Entity path prefix for the trial
        (e.g. "BAA01/trials/Baseline_Walk01").
    time_range : (start, end) or None
        Time range in seconds to extract. None = all data.

    Returns
    -------
    (marker_names, positions, frame_rate)
        marker_names : list[str]
        positions : (n_frames, n_markers, 3) array
        frame_rate : float
    """
    conn = _get_duckdb_connection(str(rrd_path))

    try:
        # Query for marker data at the entity path
        # The rrd extension exposes Points3D data with positions as nested arrays
        entity_path = f"{entity_prefix}/markers"

        # First, get the frame rate from the timeline
        try:
            rate_result = conn.execute(f"""
                SELECT DISTINCT time
                FROM recording
                WHERE entity_path = '{entity_path}'
                ORDER BY time
                LIMIT 10
            """).fetchall()
            if len(rate_result) >= 2:
                times = sorted([r[0] for r in rate_result])
                dt = np.median(np.diff(times))
                frame_rate = 1.0 / dt if dt > 0 else 200.0
            else:
                frame_rate = 200.0  # default rat mocap rate
        except Exception:
            frame_rate = 200.0

        # Query marker positions
        time_filter = ""
        if time_range:
            time_filter = f"AND time >= {time_range[0]} AND time <= {time_range[1]}"

        result = conn.execute(f"""
            SELECT time, positions, labels
            FROM recording
            WHERE entity_path = '{entity_path}'
            {time_filter}
            ORDER BY time
        """)
        rows = result.fetchall()

        if not rows:
            logger.warning(f"No marker data found at {entity_path}")
            return [], np.array([]).reshape(0, 0, 3), frame_rate

        # Parse the results
        # The rrd extension stores Points3D with positions as a flat array
        # and labels as a string array
        times = []
        all_positions = []
        marker_names = []

        for row in rows:
            t = row[0]
            positions = row[1]  # This is the positions data from Points3D
            labels = row[2]     # Marker names

            times.append(t)

            # Parse positions — format depends on rrd extension version
            if isinstance(positions, (list, tuple)):
                pos = np.array(positions, dtype=np.float64)
            elif hasattr(positions, 'as_py'):
                pos = np.array(positions.as_py(), dtype=np.float64)
            else:
                pos = np.array(positions, dtype=np.float64)

            # Reshape to (n_markers, 3) if needed
            if pos.ndim == 1:
                pos = pos.reshape(-1, 3)

            all_positions.append(pos)

            # Extract marker names from first frame
            if not marker_names and labels is not None:
                if isinstance(labels, (list, tuple)):
                    marker_names = [str(l) for l in labels]
                elif hasattr(labels, 'as_py'):
                    marker_names = [str(l) for l in labels.as_py()]

        n_frames = len(all_positions)
        n_markers = len(marker_names) if marker_names else all_positions[0].shape[0]

        # Stack to (n_frames, n_markers, 3)
        positions = np.stack(all_positions, axis=0)

        if not marker_names:
            marker_names = [f"M{i+1}" for i in range(n_markers)]

        logger.info(f"Extracted {n_frames} frames, {n_markers} markers from {entity_path}")
        return marker_names, positions, frame_rate

    finally:
        conn.close()


# -----------------------------------------------------------------------
# Force plate extraction from .rrd
# -----------------------------------------------------------------------

def extract_force_plates_from_rrd(
    rrd_path: str | Path,
    entity_prefix: str,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Extract force plate data from a Rerun .rrd recording.

    Parameters
    ----------
    rrd_path : str or Path
        Path to the .rrd file.
    entity_prefix : str
        Entity path prefix for the trial.
    time_range : (start, end) or None
        Time range in seconds to extract.

    Returns
    -------
    list of dicts with keys: 'name', 'force', 'moment', 'cop', 'times'
    Each array is (n_frames, 3).
    """
    conn = _get_duckdb_connection(str(rrd_path))

    try:
        # Find all force plate entity paths
        fp_paths_result = conn.execute(f"""
            SELECT DISTINCT entity_path
            FROM recording
            WHERE entity_path LIKE '{entity_prefix}/force_plates/%'
            ORDER BY entity_path
        """).fetchall()

        if not fp_paths_result:
            logger.warning(f"No force plate data found under {entity_prefix}/force_plates/")
            return []

        # Group by force plate name
        fp_names = set()
        for (path,) in fp_paths_result:
            parts = path.split("/")
            if len(parts) >= 5:  # .../force_plates/{fp_name}/{component}
                fp_names.add(parts[-2])

        time_filter = ""
        if time_range:
            time_filter = f"AND time >= {time_range[0]} AND time <= {time_range[1]}"

        results = []
        for fp_name in sorted(fp_names):
            fp_base = f"{entity_prefix}/force_plates/{fp_name}"

            # Extract forces
            forces = _extract_arrows3d(conn, f"{fp_base}/force", time_filter)
            moments = _extract_arrows3d(conn, f"{fp_base}/moment", time_filter)
            cops = _extract_points3d(conn, f"{fp_base}/cop", time_filter)

            if forces is not None:
                n = min(len(forces["data"]), len(moments["data"]) if moments else 0,
                        len(cops["data"]) if cops else 0)
                results.append({
                    "name": fp_name,
                    "force": forces["data"][:n],
                    "moment": moments["data"][:n] if moments else np.zeros((n, 3)),
                    "cop": cops["data"][:n] if cops else np.zeros((n, 3)),
                    "times": forces["times"][:n],
                })
                logger.info(f"Extracted {n} frames from {fp_name}")

        return results

    finally:
        conn.close()


def _extract_arrows3d(conn, entity_path: str, time_filter: str = "") -> dict | None:
    """Extract Arrows3D data (vectors) from the rrd recording."""
    try:
        result = conn.execute(f"""
            SELECT time, vectors
            FROM recording
            WHERE entity_path = '{entity_path}'
            {time_filter}
            ORDER BY time
        """)
        rows = result.fetchall()
        if not rows:
            return None

        times = []
        data = []
        for row in rows:
            times.append(row[0])
            vec = row[1]
            if isinstance(vec, (list, tuple)):
                data.append(np.array(vec, dtype=np.float64).flatten()[:3])
            elif hasattr(vec, 'as_py'):
                data.append(np.array(vec.as_py(), dtype=np.float64).flatten()[:3])
            else:
                data.append(np.array(vec, dtype=np.float64).flatten()[:3])

        return {"times": np.array(times), "data": np.array(data)}
    except Exception as e:
        logger.debug(f"Could not extract Arrows3D from {entity_path}: {e}")
        return None


def _extract_points3d(conn, entity_path: str, time_filter: str = "") -> dict | None:
    """Extract Points3D data (positions) from the rrd recording."""
    try:
        result = conn.execute(f"""
            SELECT time, positions
            FROM recording
            WHERE entity_path = '{entity_path}'
            {time_filter}
            ORDER BY time
        """)
        rows = result.fetchall()
        if not rows:
            return None

        times = []
        data = []
        for row in rows:
            times.append(row[0])
            pos = row[1]
            if isinstance(pos, (list, tuple)):
                data.append(np.array(pos, dtype=np.float64).flatten()[:3])
            elif hasattr(pos, 'as_py'):
                data.append(np.array(pos.as_py(), dtype=np.float64).flatten()[:3])
            else:
                data.append(np.array(pos, dtype=np.float64).flatten()[:3])

        return {"times": np.array(times), "data": np.array(data)}
    except Exception as e:
        logger.debug(f"Could not extract Points3D from {entity_path}: {e}")
        return None


# -----------------------------------------------------------------------
# .rrd → TRC
# -----------------------------------------------------------------------

def rrd_to_trc(
    rrd_path: str | Path,
    entity_prefix: str,
    output_dir: str | Path,
    time_range: tuple[float, float] | None = None,
    cutoff: float = 15.0,
    output_name: str | None = None,
) -> Path:
    """Extract markers from .rrd and write an OpenSim TRC file.

    Parameters
    ----------
    rrd_path : str or Path
        Path to the .rrd file.
    entity_prefix : str
        Entity path prefix (e.g. "BAA01/trials/Baseline_Walk01").
    output_dir : str or Path
        Directory to write the TRC file.
    time_range : (start, end) or None
        Time range in seconds. None = all data.
    cutoff : float
        Lowpass filter cutoff (Hz). -1 to skip filtering.
    output_name : str or None
        Output filename (default: derived from entity_prefix).

    Returns
    -------
    Path to the written .trc file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        trial_name = entity_prefix.split("/")[-1]
        output_name = f"{trial_name}.trc"
    trc_path = output_dir / output_name

    # Extract markers from .rrd
    marker_names, positions, frame_rate = extract_markers_from_rrd(
        rrd_path, entity_prefix, time_range
    )

    if positions.size == 0:
        logger.warning(f"No marker data extracted for {entity_prefix}")
        return trc_path

    n_frames, n_markers, _ = positions.shape

    # Filter markers (matching MATLAB struct2trc.m)
    if cutoff > 0:
        from scipy.signal import butter, filtfilt

        Wn = cutoff / (frame_rate / 2)
        if Wn < 1.0:
            b, a = butter(2, Wn, btype="low")
            for m_idx in range(n_markers):
                m_data = positions[:, m_idx, :]
                exists = ~np.all(m_data == 0, axis=1)
                if np.sum(exists) > 4:
                    try:
                        positions[exists, m_idx, :] = filtfilt(b, a, m_data[exists, :])
                    except Exception as e:
                        logger.warning(f"Error filtering marker {marker_names[m_idx]}: {e}")

    # Rotate to OpenSim coordinate system
    for m_idx in range(n_markers):
        positions[:, m_idx, :] = (VICON_TO_OPENSIM @ positions[:, m_idx, :].T).T

    # Write TRC file
    _write_trc_file(trc_path, positions, marker_names, frame_rate, n_frames)

    logger.info(f"Wrote TRC: {trc_path} ({n_markers} markers, {n_frames} frames)")
    return trc_path


# -----------------------------------------------------------------------
# .rrd → FP MOT + External Loads XML
# -----------------------------------------------------------------------

def rrd_to_fp_mot(
    rrd_path: str | Path,
    entity_prefix: str,
    events: GaitEvents,
    output_dir: str | Path,
    body_names: dict[str, str] | None = None,
    time_range: tuple[float, float] | None = None,
    lowpass_cutoff: float = 50.0,
    notch_low: float = 58.0,
    notch_high: float = 62.0,
    filter_data: bool = True,
    output_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Extract force plate data from .rrd and write OpenSim MOT + external loads XML.

    Parameters
    ----------
    rrd_path : str or Path
        Path to the .rrd file.
    entity_prefix : str
        Entity path prefix (e.g. "BAA01/trials/Baseline_Walk01").
    events : GaitEvents
        Gait events for determining the gait cycle window.
    output_dir : str or Path
        Directory to write output files.
    body_names : dict[str, str] or None
        Mapping of context to body name. Default: {"Left": "foot_l", "Right": "foot_r"}.
    time_range : (start, end) or None
        Time range in seconds. None = all data.
    lowpass_cutoff : float
        Lowpass filter cutoff (Hz). Default 50.
    notch_low, notch_high : float
        Notch filter band (Hz). Default 58-62.
    filter_data : bool
        Whether to apply filters.
    output_prefix : str or None
        Output filename prefix (default: derived from entity_prefix).

    Returns
    -------
    (mot_path, ext_loads_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if body_names is None:
        body_names = {"Left": "foot_l", "Right": "foot_r"}

    if output_prefix is None:
        output_prefix = entity_prefix.split("/")[-1]

    mot_path = output_dir / f"{output_prefix}_FP.mot"
    ext_loads_path = output_dir / f"{output_prefix}_ext_loads.xml"

    # Extract force plate data from .rrd
    fp_data = extract_force_plates_from_rrd(rrd_path, entity_prefix, time_range)

    if not fp_data:
        logger.warning(f"No force plate data extracted for {entity_prefix}")
        return mot_path, ext_loads_path

    # Get frame rate from the first force plate
    fp_times = fp_data[0]["times"]
    if len(fp_times) >= 2:
        analog_rate = 1.0 / np.median(np.diff(fp_times))
    else:
        analog_rate = 1000.0  # default

    frame_ratio = analog_rate / 200.0  # assume 200 Hz camera rate

    # Process each force plate
    processed_fps = []
    for fp in fp_data:
        forces = fp["force"]
        moments = fp["moment"]
        cops = fp["cop"]
        fp_name = fp["name"]

        # Apply filters (matching MATLAB struct2fp.m)
        if filter_data:
            from scipy.signal import butter, filtfilt as scipy_filtfilt

            # Notch filter (58-62 Hz)
            Wn_notch = [notch_low / (analog_rate / 2), notch_high / (analog_rate / 2)]
            if Wn_notch[0] < 1.0 and Wn_notch[1] < 1.0:
                b_notch, a_notch = butter(2, Wn_notch, btype="bandstop")
                forces = scipy_filtfilt(b_notch, a_notch, forces, axis=0)
                moments = scipy_filtfilt(b_notch, a_notch, moments, axis=0)

            # Lowpass filter (50 Hz)
            Wn_low = lowpass_cutoff / (analog_rate / 2)
            if Wn_low < 1.0:
                b_low, a_low = butter(2, Wn_low, btype="low")
                forces = scipy_filtfilt(b_low, a_low, forces, axis=0)
                moments = scipy_filtfilt(b_low, a_low, moments, axis=0)

        # Negate forces and moments (reaction → applied)
        osim_force = -forces
        osim_moment = -moments / 1000.0  # Nmm → Nm
        osim_cop = cops / 1000.0  # mm → m

        # Zero outside gait cycle
        # Determine context from force plate name
        context = "Right"  # default
        if "left" in fp_name.lower() or fp_name.lower().endswith("_l"):
            context = "Left"

        if context.lower() == "left":
            strikes = events.left_foot_strike
            offs = events.left_foot_off
        else:
            strikes = events.right_foot_strike
            offs = events.right_foot_off

        if strikes and offs:
            margin = 5
            window_start = int((strikes[0] - 1) * frame_ratio) - margin
            window_end = int((offs[-1] - 1) * frame_ratio) + margin
            osim_force[:max(0, window_start), :] = 0
            osim_force[min(len(osim_force), window_end + 1):, :] = 0
            osim_moment[:max(0, window_start), :] = 0
            osim_moment[min(len(osim_moment), window_end + 1):, :] = 0
            osim_cop[:max(0, window_start), :] = 0
            osim_cop[min(len(osim_cop), window_end + 1):, :] = 0

        body = body_names.get(context, "foot_r")

        processed_fps.append({
            "name": fp_name,
            "force": osim_force,
            "moment": osim_moment,
            "cop": osim_cop,
            "body": body,
            "context": context,
        })

    # Build MOT data matrix
    n_analog = min(len(fp["force"]) for fp in processed_fps)
    time_col = np.arange(n_analog, dtype=np.float64) / analog_rate

    mot_data = [time_col]
    column_labels = ["time"]

    for fp in processed_fps:
        i = fp["name"][-1]  # "1", "2", etc.
        mot_data.extend([
            fp["force"][:n_analog, 0], fp["force"][:n_analog, 1], fp["force"][:n_analog, 2],
            fp["cop"][:n_analog, 0], fp["cop"][:n_analog, 1], fp["cop"][:n_analog, 2],
            fp["moment"][:n_analog, 0], fp["moment"][:n_analog, 1], fp["moment"][:n_analog, 2],
        ])
        column_labels.extend([
            f"force{i}_vx", f"force{i}_vy", f"force{i}_vz",
            f"force{i}_px", f"force{i}_py", f"force{i}_pz",
            f"moment{i}_x", f"moment{i}_y", f"moment{i}_z",
        ])

    mot_matrix = np.column_stack(mot_data)

    # Write MOT file
    _write_mot_file(mot_path, mot_matrix, column_labels)

    # Write external loads XML
    _write_external_loads_xml(ext_loads_path, processed_fps, f"{output_prefix}_FP.mot")

    logger.info(f"Wrote FP MOT: {mot_path} ({len(processed_fps)} plates, {n_analog} frames)")
    logger.info(f"Wrote ext loads: {ext_loads_path}")

    return mot_path, ext_loads_path


# -----------------------------------------------------------------------
# File writers
# -----------------------------------------------------------------------

def _write_trc_file(
    filepath: Path,
    marker_data: np.ndarray,
    marker_names: list[str],
    frame_rate: float,
    num_frames: int,
    units: str = "mm",
) -> None:
    """Write a TRC file in OpenSim format."""
    n_markers = len(marker_names)
    frame_col = np.arange(1, num_frames + 1).reshape(-1, 1)
    time_col = (frame_col - 1) / frame_rate
    data_cols = marker_data.reshape(num_frames, n_markers * 3)
    trc_data = np.hstack([frame_col, time_col, data_cols])

    with open(filepath, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{frame_rate}\t{frame_rate}\t{num_frames}\t{n_markers}\t{units}\t"
                f"{frame_rate}\t1\t{num_frames}\n")
        f.write("Frame#\tTime\t")
        for name in marker_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")
        f.write("\t\t")
        for i in range(n_markers):
            f.write(f"X{i + 1}\tY{i + 1}\tZ{i + 1}\t")
        f.write("\n\n")
        for row in trc_data:
            f.write(f"{int(row[0])}\t{row[1]:.6f}\t")
            for val in row[2:]:
                f.write(f"{val:.6f}\t")
            f.write("\n")


def _write_mot_file(filepath: Path, data: np.ndarray, column_labels: list[str]) -> None:
    """Write a MOT file in OpenSim format."""
    n_rows, n_cols = data.shape
    with open(filepath, "w") as f:
        f.write(f"{filepath.name}\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={n_cols}\n")
        f.write("endheader\n")
        f.write("\t".join(column_labels) + "\n")
        for row in data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")


def _write_external_loads_xml(
    filepath: Path,
    fp_data: list[dict],
    data_file: str,
) -> None:
    """Write an OpenSim ExternalLoads XML file."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<OpenSimDocument Version="40000">',
        '  <ExternalLoads>',
        f'    <datafile>{data_file}</datafile>',
    ]
    for fp in fp_data:
        i = fp["name"][-1]
        lines.extend([
            '    <objects>',
            '      <ExternalForce>',
            f'        <name>FP{i}</name>',
            f'        <applied_to_body>{fp["body"]}</applied_to_body>',
            '        <force_expressed_in_body>ground</force_expressed_in_body>',
            '        <point_expressed_in_body>ground</point_expressed_in_body>',
            f'        <force_identifier>force{i}_v</force_identifier>',
            f'        <point_identifier>force{i}_p</point_identifier>',
            f'        <torque_identifier>moment{i}_</torque_identifier>',
            f'        <data_source_name>{data_file}</data_source_name>',
            '      </ExternalForce>',
            '    </objects>',
        ])
    lines.extend([
        '  </ExternalLoads>',
        '</OpenSimDocument>',
    ])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n")
