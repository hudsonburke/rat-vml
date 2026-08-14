"""Default configuration values for rat-vml.

These values are also defined in params.yaml for DVC pipelines,
but are hardcoded here for direct script execution.
"""

import numpy as np

# Vicon -> OpenSim coordinate rotation
# Vicon: X forward, Y left, Z up
# OpenSim: X right, Y up, Z forward (toward direction of travel)
VICON_TO_OPENSIM = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)

# Markers required for IK/analysis
REQUIRED_MARKERS = [
    "TAIL",
    "SPL6",
    "LASI",
    "RASI",
    "LHIP",
    "LKNE",
    "LANK",
    "LTOE",
    "RHIP",
    "RKNE",
    "RANK",
    "RTOE",
]

# Marker weights for IK (knee markers reduced to 0.1)
IK_MARKER_WEIGHTS = {
    "RKNE": 0.1,
    "LKNE": 0.1,
}

# Default filter cutoff frequency (Hz)
DEFAULT_CUTOFF_HZ = 15.0
