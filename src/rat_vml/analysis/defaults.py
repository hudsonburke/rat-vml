"""Rat-specific default values for analysis pipeline.

These constants define the coordinate names, marker sets, and file paths
used by the rat hindlimb model analysis workflows.  If your model differs,
change these at import time::

    from rat_vml.analysis import defaults
    defaults.COORD_NAMES = ["my_coord1", ...]
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Coordinate names (matching the bilateral rat model)
# ---------------------------------------------------------------------------
COORD_NAMES = [
    "hip_r_flx", "hip_r_add", "hip_r_int",
    "knee_r_flx",
    "ankle_r_flx", "ankle_r_add", "ankle_r_int",
    "hip_l_flx", "hip_l_add", "hip_l_int",
    "knee_l_flx",
    "ankle_l_flx", "ankle_l_add", "ankle_l_int",
]

COORD_NAMES_R = [
    "hip_r_flx", "hip_r_add", "hip_r_int",
    "knee_r_flx",
    "ankle_r_flx", "ankle_r_add", "ankle_r_int",
]

# ---------------------------------------------------------------------------
# Joint moment names (OpenSim ID output column names)
# ---------------------------------------------------------------------------
MOMENT_NAMES_R = [
    "hip_r_flx_moment", "hip_r_add_moment", "hip_r_int_moment",
    "knee_r_flx_moment",
    "ankle_r_flx_moment",
]

# ---------------------------------------------------------------------------
# Default file paths (relative to the rat-hindlimb-model repo root)
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models/osim")
BILATERAL_MODEL = MODEL_DIR / "rat_hindlimb_bilateral.osim"
UNILATERAL_MODEL = MODEL_DIR / "rat_hindlimb_unilateral.osim"
MARKER_SET_PATH = MODEL_DIR / "xml" / "rat_hindlimb_bilateral_markers.xml"
SCALE_SETUP_PATH = MODEL_DIR / "xml" / "rat_hindlimb_bilateral_scale_setup.xml"

# ---------------------------------------------------------------------------
# Rat-specific constants
# ---------------------------------------------------------------------------
# Base femur/tibia lengths (mm) from the unscaled model — used by scaling
BASE_FEMUR_LENGTH_MM = 32.0    # approximate, from model geometry
BASE_TIBIA_LENGTH_MM = 39.0    # approximate, from model geometry

# Marker names grouped by function
MARKER_NAMES = [
    "L6", "Caudal5",
    "R_AntIliacCrest", "L_AntIliacCrest",
    "R_GreaterTrochanter", "L_GreaterTrochanter",
    "R_LatFemoralEpicondyle", "L_LatFemoralEpicondyle",
    "R_LatMalleolus", "L_LatMalleolus",
    "R_Distal5thMetatarsal", "L_Distal5thMetatarsal",
]
