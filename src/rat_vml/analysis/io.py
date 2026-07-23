"""C3D to OpenSim file format conversion.

Mirrors the MATLAB struct2trc.m and struct2fp.m pipelines from
UVA-MAMP-Lab/Toolbox: reads C3D files with ezc3d, applies the
Vicon→OpenSim coordinate transform and filtering, and writes
TRC (markers) and MOT (force plates) files for OpenSim IK/ID.

Usage::

    from rat_vml.analysis.io import c3d_to_trc, c3d_to_fp_mot

    # Export markers to TRC
    trc_path = c3d_to_trc("Walk01.c3d", output_dir="data/ik/")

    # Export force plates to MOT + external loads XML
    mot_path, ext_loads_path = c3d_to_fp_mot(
        "Walk01.c3d",
        output_dir="data/id/",
        events=events,  # GaitEvents object
        body_names={"Left": "foot_l", "Right": "foot_r"},
    )
"""

import logging
from pathlib import Path

import ezc3d
import numpy as np

from .events import GaitEvents, VICON_TO_OPENSIM

logger = logging.getLogger(__name__)


def _butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter (matches MATLAB filtfilt with order 2).

    Parameters
    ----------
    data : (N, C) array
    cutoff : Hz
    fs : Hz
    order : filter order (MATLAB uses 2, filtfilt doubles it to 4th order effective)
    """
    from scipy.signal import butter, filtfilt

    Wn = cutoff / (fs / 2)
    if Wn >= 1.0:
        return data  # Cutoff above Nyquist — no filtering needed
    b, a = butter(order, Wn, btype="low")
    return filtfilt(b, a, data, axis=0)


def _butter_notch(data: np.ndarray, freq_low: float, freq_high: float, fs: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-stop filter."""
    from scipy.signal import butter, filtfilt

    Wn = [freq_low / (fs / 2), freq_high / (fs / 2)]
    if Wn[0] >= 1.0 or Wn[1] >= 1.0:
        return data
    b, a = butter(order, Wn, btype="bandstop")
    return filtfilt(b, a, data, axis=0)


# -----------------------------------------------------------------------
# C3D → TRC
# -----------------------------------------------------------------------

def c3d_to_trc(
    c3d_path: str | Path,
    output_dir: str | Path,
    cutoff: float = 15.0,
    output_name: str | None = None,
) -> Path:
    """Read markers from a C3D file and write an OpenSim TRC file.

    Mirrors MATLAB struct2trc.m:
      1. Read markers from C3D
      2. Apply 2nd-order Butterworth lowpass filter (cutoff Hz)
      3. Rotate from Vicon to OpenSim coordinates
      4. Write TRC in OpenSim format

    Parameters
    ----------
    c3d_path : str or Path
        Path to the C3D file.
    output_dir : str or Path
        Directory to write the TRC file.
    cutoff : float
        Lowpass filter cutoff frequency in Hz. Use -1 to skip filtering.
    output_name : str or None
        Output filename (default: same as C3D stem with .trc extension).

    Returns
    -------
    Path to the written .trc file.
    """
    c3d_path = Path(c3d_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = c3d_path.stem + ".trc"
    trc_path = output_dir / output_name

    # Read C3D
    c3d = ezc3d.c3d(str(c3d_path))

    point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    marker_names = [str(n) for n in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    n_markers = len(marker_names)

    # Marker data: (4, n_markers, n_frames) → (n_frames, n_markers, 3)
    raw_points = c3d["data"]["points"]
    n_frames = raw_points.shape[2]
    # Extract XYZ, transpose to (n_frames, n_markers, 3)
    marker_data = np.transpose(raw_points[:3, :, :], (2, 1, 0)).astype(np.float64)

    # Filter each marker's X,Y,Z independently
    if cutoff > 0:
        for m_idx in range(n_markers):
            m_data = marker_data[:, m_idx, :]  # (n_frames, 3)
            # Only filter non-zero frames (marker exists)
            exists = ~np.all(m_data == 0, axis=1)
            if np.sum(exists) > 4:  # Need enough points to filter
                try:
                    m_data[exists, :] = _butter_lowpass(m_data[exists, :], cutoff, point_rate)
                except Exception as e:
                    logger.warning(f"Error filtering marker {marker_names[m_idx]}: {e}")

    # Rotate to OpenSim coordinate system
    for m_idx in range(n_markers):
        marker_data[:, m_idx, :] = (VICON_TO_OPENSIM @ marker_data[:, m_idx, :].T).T

    # Write TRC file
    _write_trc_file(
        trc_path,
        marker_data,
        marker_names,
        point_rate,
        n_frames,
        "mm",
    )

    logger.info(f"Wrote TRC: {trc_path} ({n_markers} markers, {n_frames} frames)")
    return trc_path


def _write_trc_file(
    filepath: Path,
    marker_data: np.ndarray,
    marker_names: list[str],
    point_rate: float,
    num_frames: int,
    units: str = "mm",
) -> None:
    """Write a TRC file in OpenSim format.

    Matches the MATLAB writeTRC.m output format.

    Parameters
    ----------
    filepath : Path
        Output .trc file path.
    marker_data : (n_frames, n_markers, 3) array
        Marker positions.
    marker_names : list[str]
        Marker names.
    point_rate : float
        Camera frame rate (Hz).
    num_frames : int
        Number of frames.
    units : str
        Coordinate units (default "mm").
    """
    n_markers = len(marker_names)

    # Build TRC data matrix: [frame#, time, x1, y1, z1, x2, y2, z2, ...]
    frame_col = np.arange(1, num_frames + 1).reshape(-1, 1)
    time_col = ((frame_col - 1) / point_rate)
    data_cols = marker_data.reshape(num_frames, n_markers * 3)
    trc_data = np.hstack([frame_col, time_col, data_cols])

    with open(filepath, "w") as f:
        # Header line
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")

        # Metadata
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{point_rate}\t{point_rate}\t{num_frames}\t{n_markers}\t{units}\t"
                f"{point_rate}\t1\t{num_frames}\n")

        # Column names row 1: Frame# Time marker1 marker2 ...
        f.write("Frame#\tTime\t")
        for name in marker_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")

        # Column names row 2: (blank) (blank) X1 Y1 Z1 X2 Y2 Z2 ...
        f.write("\t\t")
        for i in range(n_markers):
            f.write(f"X{i + 1}\tY{i + 1}\tZ{i + 1}\t")
        f.write("\n\n")

        # Data rows
        for row in trc_data:
            f.write(f"{int(row[0])}\t{row[1]:.6f}\t")
            for val in row[2:]:
                f.write(f"{val:.6f}\t")
            f.write("\n")


# -----------------------------------------------------------------------
# C3D → FP MOT + External Loads XML
# -----------------------------------------------------------------------

def c3d_to_fp_mot(
    c3d_path: str | Path,
    output_dir: str | Path,
    events: GaitEvents,
    body_names: dict[str, str] | None = None,
    lowpass_cutoff: float = 50.0,
    notch_low: float = 58.0,
    notch_high: float = 62.0,
    filter_data: bool = True,
    output_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Read force plate data from a C3D file and write OpenSim MOT + external loads XML.

    Mirrors MATLAB struct2fp.m:
      1. Read force plate data from C3D (forces, moments, COP, geometry)
      2. Apply notch filter (58-62 Hz) and lowpass filter (50 Hz)
      3. Transform from plate-local to world coordinates
      4. Calculate COP if not provided by the force plate
      5. Rotate into OpenSim coordinate system
      6. Zero out data outside the gait cycle
      7. Write MOT file with force, COP, moment columns
      8. Write ExternalLoads XML file

    Parameters
    ----------
    c3d_path : str or Path
        Path to the C3D file.
    output_dir : str or Path
        Directory to write output files.
    events : GaitEvents
        Gait events for determining the gait cycle window.
    body_names : dict[str, str] or None
        Mapping of context ("Left"/"Right") to body name ("foot_l"/"foot_r").
        Default: {"Left": "foot_l", "Right": "foot_r"}.
    lowpass_cutoff : float
        Lowpass filter cutoff (Hz). Default 50.
    notch_low, notch_high : float
        Notch filter band (Hz). Default 58-62.
    filter_data : bool
        Whether to apply filters. Default True.
    output_prefix : str or None
        Output filename prefix (default: C3D stem).

    Returns
    -------
    (mot_path, ext_loads_path)
        Paths to the written MOT and external loads XML files.
    """
    c3d_path = Path(c3d_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if body_names is None:
        body_names = {"Left": "foot_l", "Right": "foot_r"}

    if output_prefix is None:
        output_prefix = c3d_path.stem

    mot_path = output_dir / f"{output_prefix}_FP.mot"
    ext_loads_path = output_dir / f"{output_prefix}_ext_loads.xml"

    # Read C3D with force plate data extraction
    c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)

    point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    n_frames = c3d["header"]["points"]["last_frame"] - c3d["header"]["points"]["first_frame"] + 1

    # Get force plate metadata
    platforms = c3d["data"]["platform"]
    n_plates = len(platforms)

    if n_plates == 0:
        logger.warning(f"No force plate data in {c3d_path}")
        return mot_path, ext_loads_path

    # Determine analog rate from force plate data
    analog_rate = platforms[0]["force"].shape[0] / n_frames * point_rate
    frame_ratio = analog_rate / point_rate

    # Process each force plate
    all_fp_data: list[dict] = []
    ext_forces: list[dict] = []

    for p_idx in range(n_plates):
        plat = platforms[p_idx]
        fp_name = f"FP{p_idx + 1}"

        # Raw data from C3D: (n_analog_frames, 3)
        forces = np.asarray(plat["force"], dtype=np.float64)
        moments = np.asarray(plat["moment"], dtype=np.float64)
        cop = np.asarray(plat["center_of_pressure"], dtype=np.float64)

        # Geometry
        corners = np.asarray(plat["corners"], dtype=np.float64)  # (3, 4)
        origin = np.asarray(plat["origin"], dtype=np.float64)  # (3,)

        # Calculate WorldR from corners (plate-local to world)
        # The corners define the plate orientation in world coords
        # Use the first two edge vectors to form the rotation
        edge1 = corners[:, 1] - corners[:, 0]
        edge2 = corners[:, 3] - corners[:, 0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        edge1 = edge1 / np.linalg.norm(edge1)
        edge2 = np.cross(normal, edge1)
        world_r = np.column_stack([edge1, edge2, normal])
        world_t = origin.reshape(1, 3)

        # Apply filters (notch then lowpass, matching MATLAB)
        if filter_data:
            forces = _butter_notch(forces, notch_low, notch_high, analog_rate)
            forces = _butter_lowpass(forces, lowpass_cutoff, analog_rate)
            moments = _butter_notch(moments, notch_low, notch_high, analog_rate)
            moments = _butter_lowpass(moments, lowpass_cutoff, analog_rate)

        # Transform from plate-local to world coordinates
        world_force = (world_r @ forces.T).T
        world_moment = (world_r @ moments.T).T

        # Handle COP: either from C3D or calculated from force/moment
        if np.any(cop):
            world_pos = (world_r @ cop.T).T + world_t
        else:
            logger.info(f"No CoP data for {fp_name}, calculating from force/moment")
            h = world_t[0, 2]  # plate surface thickness
            with np.errstate(divide="ignore", invalid="ignore"):
                xp = (h * world_force[:, 0] - world_moment[:, 1]) / world_force[:, 2]
                yp = (h * world_force[:, 1] + world_moment[:, 0]) / world_force[:, 2]
            xp = np.where(np.isfinite(xp), xp, 0.0)
            yp = np.where(np.isfinite(yp), yp, 0.0)
            world_pos = np.column_stack([xp, yp, np.zeros_like(xp)]) + world_t

            # Free moment calculation
            adjusted_moment = np.zeros_like(world_moment)
            adjusted_moment[:, 2] = (
                world_moment[:, 2]
                + world_force[:, 0] * (world_pos[:, 1] - world_t[0, 1])
                - world_force[:, 1] * (world_pos[:, 0] - world_t[0, 0])
            )
            world_moment = adjusted_moment

        # Rotate into OpenSim coordinate system
        # Force negated: reaction force → applied force
        osim_force = -(VICON_TO_OPENSIM @ world_force.T).T
        osim_moment = -(VICON_TO_OPENSIM @ world_moment.T).T / 1000.0  # Nmm → Nm
        osim_pos = (VICON_TO_OPENSIM @ world_pos.T).T / 1000.0  # mm → m

        # Determine context (Left/Right) from force plate setup
        context = "Right"  # default
        try:
            fp_channel = c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"]
            analog_labels = [str(l) for l in c3d["parameters"]["ANALOG"]["LABELS"]["value"]]
            if p_idx < fp_channel.shape[0]:
                first_ch = int(fp_channel[p_idx, 0]) - 1
                if first_ch < len(analog_labels):
                    label = analog_labels[first_ch].lower()
                    if "left" in label or "l" == label[0]:
                        context = "Left"
        except (KeyError, IndexError):
            pass

        # Zero outside gait cycle
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
            osim_pos[:max(0, window_start), :] = 0
            osim_pos[min(len(osim_pos), window_end + 1):, :] = 0

        body = body_names.get(context, "foot_r")

        all_fp_data.append({
            "name": fp_name,
            "force": osim_force,
            "moment": osim_moment,
            "cop": osim_pos,
            "body": body,
            "context": context,
        })

    # Build MOT data: [time, force1_vx, vy, vz, force1_px, py, pz, moment1_x, y, z, ...]
    n_analog = int(n_frames * frame_ratio)
    time_col = np.arange(n_analog, dtype=np.float64) / analog_rate

    mot_data = [time_col]
    column_labels = ["time"]

    for fp in all_fp_data:
        i = fp["name"][-1]  # "1", "2", etc.
        mot_data.extend([
            fp["force"][:, 0], fp["force"][:, 1], fp["force"][:, 2],
            fp["cop"][:, 0], fp["cop"][:, 1], fp["cop"][:, 2],
            fp["moment"][:, 0], fp["moment"][:, 1], fp["moment"][:, 2],
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
    _write_external_loads_xml(ext_loads_path, all_fp_data, f"{output_prefix}_FP.mot")

    logger.info(f"Wrote FP MOT: {mot_path} ({n_plates} plates, {n_analog} frames)")
    logger.info(f"Wrote ext loads: {ext_loads_path}")

    return mot_path, ext_loads_path


def _write_mot_file(filepath: Path, data: np.ndarray, column_labels: list[str]) -> None:
    """Write a MOT file in OpenSim format.

    Matches the MATLAB writeMOT.m output format.
    """
    n_rows, n_cols = data.shape

    with open(filepath, "w") as f:
        # Header
        f.write(f"{filepath.name}\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={n_cols}\n")
        f.write("endheader\n")

        # Column labels
        f.write("\t".join(column_labels) + "\n")

        # Data rows
        for row in data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")


def _write_external_loads_xml(
    filepath: Path,
    fp_data: list[dict],
    data_file: str,
) -> None:
    """Write an OpenSim ExternalLoads XML file.

    Parameters
    ----------
    filepath : Path
        Output XML file path.
    fp_data : list[dict]
        Force plate data dicts with 'name', 'body', 'context' keys.
    data_file : str
        Name of the MOT data file (relative path).
    """
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
