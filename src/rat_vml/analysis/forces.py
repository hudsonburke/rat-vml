"""Force plate data processing.

Mirrors the MATLAB struct2fp.m pipeline:
- Transform force plate data from local to world coordinates
- Apply 4th-order Butterworth lowpass (50 Hz) and notch filter (58-62 Hz)
- Rotate into OpenSim coordinate system
- Zero out data outside the gait cycle
- Write MOT file and external loads XML
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Vicon → OpenSim coordinate system rotation
VICON_TO_OPENSIM = np.array([
    [1, 0,  0],
    [0, 0,  1],
    [0, -1, 0],
], dtype=float)


def _butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    sample_rate: float,
    order: int = 2,
) -> np.ndarray:
    """Apply zero-phase Butterworth lowpass filter.

    Parameters
    ----------
    data : (N, C) array
    cutoff : Hz
    sample_rate : Hz
    order : filter order (MATLAB uses order 2 with filtfilt = 4th order effective)
    """
    from scipy.signal import butter, filtfilt as scipy_filtfilt

    nyq = sample_rate / 2.0
    Wn = cutoff / nyq
    b, a = butter(order, Wn, btype="low")
    return scipy_filtfilt(b, a, data, axis=0)


def _butter_notch_filter(
    data: np.ndarray,
    freq_low: float,
    freq_high: float,
    sample_rate: float,
    order: int = 2,
) -> np.ndarray:
    """Apply zero-phase Butterworth notch (band-stop) filter.

    Parameters
    ----------
    data : (N, C) array
    freq_low, freq_high : Hz band to reject
    sample_rate : Hz
    order : filter order
    """
    from scipy.signal import butter, filtfilt as scipy_filtfilt

    nyq = sample_rate / 2.0
    Wn = [freq_low / nyq, freq_high / nyq]
    b, a = butter(order, Wn, btype="bandstop")
    return scipy_filtfilt(b, a, data, axis=0)


def process_force_plate(
    force_data: np.ndarray,
    moment_data: np.ndarray,
    cop_data: np.ndarray,
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
    sample_rate: float,
    filter_data: bool = True,
    lowpass_cutoff: float = 50.0,
    notch_low: float = 58.0,
    notch_high: float = 62.0,
) -> dict[str, np.ndarray]:
    """Process a single force plate's data.

    Parameters
    ----------
    force_data : (N, 3) array
        Force in plate-local coordinates.
    moment_data : (N, 3) array
        Moment in plate-local coordinates.
    cop_data : (N, 3) array
        Center of pressure in plate-local coordinates (may be zeros).
    world_rotation : (3, 3) array
        Rotation matrix from plate-local to world coordinates.
    world_translation : (3,) or (1, 3) array
        Translation from plate-local to world coordinates.
    sample_rate : float
        Sampling rate (Hz).
    filter_data : bool
        Whether to apply lowpass + notch filters.
    lowpass_cutoff : float
        Lowpass filter cutoff (Hz). Default 50.
    notch_low, notch_high : float
        Notch filter band (Hz). Default 58-62 Hz.

    Returns
    -------
    dict with keys: 'force', 'moment', 'cop' (all in OpenSim coordinates)
    """
    # Transform from plate-local to world coordinates
    world_force = (world_rotation @ force_data.T).T
    world_moment = (world_rotation @ moment_data.T).T
    world_pos = (world_rotation @ cop_data.T).T + world_translation

    # Calculate CoP from force and moment data if not provided
    if not np.any(cop_data):
        h = world_translation[2] if len(world_translation.shape) == 1 else world_translation[0, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xp = (h * world_force[:, 0] - world_moment[:, 1]) / world_force[:, 2]
            yp = (h * world_force[:, 1] + world_moment[:, 0]) / world_force[:, 2]
        xp = np.where(np.isfinite(xp), xp, 0.0)
        yp = np.where(np.isfinite(yp), yp, 0.0)
        world_pos = np.column_stack([xp, yp, np.zeros_like(xp)])
        # Recalculate moments as free moments
        adjusted_moment = np.zeros_like(world_moment)
        adjusted_moment[:, 2] = (
            world_moment[:, 2]
            + world_force[:, 0] * world_pos[:, 1]
            - world_force[:, 1] * world_pos[:, 0]
        )
        world_moment = adjusted_moment
        world_pos = world_pos + world_translation

    # Apply filters (matching MATLAB: notch then lowpass)
    if filter_data:
        world_force = _butter_notch_filter(world_force, notch_low, notch_high, sample_rate)
        world_force = _butter_lowpass_filter(world_force, lowpass_cutoff, sample_rate)
        world_moment = _butter_notch_filter(world_moment, notch_low, notch_high, sample_rate)
        world_moment = _butter_lowpass_filter(world_moment, lowpass_cutoff, sample_rate)
        # Don't filter position data

    # Rotate into OpenSim coordinate system
    # Force is negated (reaction force → applied force)
    osim_force = -(VICON_TO_OPENSIM @ world_force.T).T
    osim_moment = -(VICON_TO_OPENSIM @ world_moment.T).T / 1000.0  # Nmm → Nm
    osim_pos = (VICON_TO_OPENSIM @ world_pos.T).T / 1000.0  # mm → m

    return {
        "force": osim_force,
        "moment": osim_moment,
        "cop": osim_pos,
    }


def zero_outside_gait_cycle(
    data: np.ndarray,
    foot_strikes: list[int],
    foot_offs: list[int],
    frame_ratio: float,
    margin: int = 5,
) -> np.ndarray:
    """Zero out force plate data outside the gait cycle window.

    Mirrors the MATLAB logic from struct2fp.m: zero data outside the
    window from the first foot strike to the last foot off.

    Parameters
    ----------
    data : (N, C) array
        Data to zero out.
    foot_strikes : list[int]
        Foot strike frame numbers (1-indexed, camera frames).
    foot_offs : list[int]
        Foot off frame numbers (1-indexed, camera frames).
    frame_ratio : float
        Ratio of force plate rate to camera rate.
    margin : int
        Extra frames around the window to keep.
    """
    if not foot_strikes or not foot_offs:
        return data

    window_start = int((foot_strikes[0] - 1) * frame_ratio) - margin
    window_end = int((foot_offs[-1] - 1) * frame_ratio) + margin

    result = data.copy()
    result[:max(0, window_start), :] = 0
    result[min(len(data), window_end + 1):, :] = 0

    return result


def write_external_loads_xml(
    filepath: Path,
    fp_name: str,
    body_name: str,
    data_file: str,
) -> Path:
    """Write an OpenSim ExternalLoads XML file.

    Parameters
    ----------
    filepath : Path
        Output XML file path.
    fp_name : str
        Force plate name (used as ExternalForce name).
    body_name : str
        Body to which the force is applied ("foot_l" or "foot_r").
    data_file : str
        Name of the MOT data file (not full path).
    """
    import opensim as osim

    ext_loads = osim.ExternalLoads()
    ext_force = osim.ExternalForce()
    ext_force.setName(fp_name)
    ext_force.setAppliedToBodyName(body_name)
    ext_force.setForceExpressedInBodyName("ground")
    ext_force.setPointExpressedInBodyName("ground")
    ext_force.setForceIdentifier(f"force{fp_name}_v")
    ext_force.setPointIdentifier(f"force{fp_name}_p")
    ext_force.setTorqueIdentifier(f"moment{fp_name}_")
    ext_force.set_data_source_name(data_file)
    ext_loads.cloneAndAppend(ext_force)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ext_loads.printToXML(str(filepath))
    return filepath
