"""Trial validation and gait event handling.

Mirrors the MATLAB pipeline's event detection and trial filtering logic
from the UVA-MAMP-Lab Rats/Toolbox repos.

The expected gait event sequence for a valid walking trial is:
    FootStrike1, FootOff1, FootStrike2, FootOff2,
    FootStrike3, FootOff3, FootStrike4

(4 foot strikes and 3 foot offs alternating, same side)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Vicon → OpenSim coordinate system rotation (90° about X-axis)
# Matches the MATLAB: R = [1 0 0; 0 0 1; 0 -1 0]
VICON_TO_OPENSIM = np.array([
    [1, 0,  0],
    [0, 0,  1],
    [0, -1, 0],
], dtype=float)


@dataclass
class GaitEvents:
    """Gait events from a walking trial.

    All values are frame numbers (1-indexed, matching Vicon convention).
    """
    left_foot_strike: list[int]
    left_foot_off: list[int]
    right_foot_strike: list[int]
    right_foot_off: list[int]
    total_frames: int
    frame_rate: float

    def to_times(self) -> dict[str, list[float]]:
        """Convert frame numbers to times (seconds)."""
        return {
            "left_foot_strike": [(f - 1) / self.frame_rate for f in self.left_foot_strike],
            "left_foot_off": [(f - 1) / self.frame_rate for f in self.left_foot_off],
            "right_foot_strike": [(f - 1) / self.frame_rate for f in self.right_foot_strike],
            "right_foot_off": [(f - 1) / self.frame_rate for f in self.right_foot_off],
        }

    @property
    def has_events(self) -> bool:
        """True if the trial has any foot strike or foot off events."""
        return any([
            self.left_foot_strike, self.left_foot_off,
            self.right_foot_strike, self.right_foot_off,
        ])


def extract_events_from_c3d(c3d_path: str | Path) -> GaitEvents:
    """Extract gait events from a C3D file.

    Reads events from the C3D file's EVENT group (if present) or from
    the .Trial.enf metadata file adjacent to the C3D.

    Parameters
    ----------
    c3d_path : str or Path
        Path to a C3D file.

    Returns
    -------
    GaitEvents
        Extracted events with frame numbers and frame rate.
    """
    import ezc3d

    c3d = ezc3d.c3d(str(c3d_path))
    point_rate = c3d["header"]["points"]["frame_rate"]
    num_frames = c3d["header"]["points"]["last_frame"] - c3d["header"]["points"]["first_frame"] + 1

    # Try C3D EVENT group first
    events = GaitEvents(
        left_foot_strike=[], left_foot_off=[],
        right_foot_strike=[], right_foot_off=[],
        total_frames=num_frames,
        frame_rate=point_rate,
    )

    if "EVENT" in c3d["parameters"]:
        ev_params = c3d["parameters"]["EVENT"]
        if "USED" in ev_params and ev_params["USED"]["value"][0] > 0:
            contexts = ev_params["CONTEXTS"]["value"]
            labels = ev_params["LABELS"]["value"]
            times = ev_params["TIMES"]["value"]  # [2, N] array

            for i in range(len(labels)):
                context = contexts[i].strip().lower()
                label = labels[i].strip().lower()
                time = times[0, i] + times[1, i] / 60.0  # seconds
                frame = int(round(time * point_rate)) + 1  # 1-indexed

                if "left" in context:
                    if "strike" in label or "foot" in label and "off" not in label:
                        events.left_foot_strike.append(frame)
                    elif "off" in label:
                        events.left_foot_off.append(frame)
                elif "right" in context:
                    if "strike" in label or "foot" in label and "off" not in label:
                        events.right_foot_strike.append(frame)
                    elif "off" in label:
                        events.right_foot_off.append(frame)

    # Sort events by frame number
    events.left_foot_strike.sort()
    events.left_foot_off.sort()
    events.right_foot_strike.sort()
    events.right_foot_off.sort()

    return events


def extract_events_from_enf(enf_path: str | Path, frame_rate: float, total_frames: int) -> GaitEvents:
    """Extract gait events from a Vicon .Trial.enf file.

    .enf files have a format like:
        [Events]
        LeftFootStrike=123,456,789
        LeftFootOff=234,567
        RightFootStrike=100,400,700
        RightFootOff=200,500

    Parameters
    ----------
    enf_path : str or Path
        Path to a .Trial.enf file.
    frame_rate : float
        Camera frame rate (Hz).
    total_frames : int
        Total number of frames in the trial.

    Returns
    -------
    GaitEvents
        Extracted events with frame numbers and frame rate.
    """
    events = GaitEvents(
        left_foot_strike=[], left_foot_off=[],
        right_foot_strike=[], right_foot_off=[],
        total_frames=total_frames,
        frame_rate=frame_rate,
    )

    enf_path = Path(enf_path)
    if not enf_path.exists():
        return events

    text = enf_path.read_text(errors="replace")
    current_section = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue

        if current_section.lower() == "events" and "=" in line:
            key, val = line.split("=", 1)
            key = key.strip().lower()
            val = val.strip().rstrip("\r")

            frames = []
            for part in val.split(","):
                part = part.strip()
                if part:
                    try:
                        frames.append(int(float(part)))
                    except ValueError:
                        pass

            if "left" in key:
                if "strike" in key or "foot" in key:
                    events.left_foot_strike.extend(frames)
                elif "off" in key:
                    events.left_foot_off.extend(frames)
            elif "right" in key:
                if "strike" in key or "foot" in key:
                    events.right_foot_strike.extend(frames)
                elif "off" in key:
                    events.right_foot_off.extend(frames)

    events.left_foot_strike.sort()
    events.left_foot_off.sort()
    events.right_foot_strike.sort()
    events.right_foot_off.sort()

    return events


def validate_walking_trial(
    events: GaitEvents,
    side: str = "right",
    min_events: int = 7,
) -> tuple[bool, str]:
    """Validate that a walking trial has the expected gait event sequence.

    The expected pattern is 4 foot strikes and 3 foot offs alternating
    on the same side, plus at least 1 event on the contralateral side.

    Parameters
    ----------
    events : GaitEvents
        The trial's gait events.
    side : str
        Primary analysis side ("left" or "right").
    min_events : int
        Minimum number of events required on the primary side.

    Returns
    -------
    (is_valid, reason)
        True if the trial passes validation, with a reason string if not.
    """
    if side == "right":
        strikes = events.right_foot_strike
        offs = events.right_foot_off
    else:
        strikes = events.left_foot_strike
        offs = events.left_foot_off

    if len(strikes) == 0 and len(offs) == 0:
        return False, "no events found"

    # Check minimum event count
    total = len(strikes) + len(offs)
    if total < min_events:
        return False, f"only {total} events (need {min_events})"

    # Check alternation: strikes and offs should interleave
    # Expected: S, O, S, O, S, O, S (4 strikes, 3 offs)
    if len(strikes) < 4:
        return False, f"only {len(strikes)} strikes (need ≥4)"

    if len(offs) < 3:
        return False, f"only {len(offs)} offs (need ≥3)"

    # Check ordering: all events should be increasing
    all_events = sorted(strikes + offs)
    if all_events != sorted(all_events):
        return False, "events not in chronological order"

    # Check that first event is a strike and last is a strike
    if strikes[0] > offs[0]:
        return False, "first event should be a foot strike"

    if strikes[-1] < offs[-1]:
        return False, "last event should be a foot strike"

    return True, "valid"


def get_gait_cycle_times(events: GaitEvents, side: str = "right") -> dict:
    """Extract stance, swing, and gait cycle time windows.

    Parameters
    ----------
    events : GaitEvents
        Trial events.
    side : str
        Side to analyze ("left" or "right").

    Returns
    -------
    dict with keys: 'stance', 'swing', 'gait_cycle', each a (start, end) tuple in seconds.
    """
    times = events.to_times()

    if side == "right":
        strikes = times["right_foot_strike"]
        offs = times["right_foot_off"]
    else:
        strikes = times["left_foot_strike"]
        offs = times["left_foot_off"]

    # Full gait cycle: first foot strike to last foot strike
    gait_cycle = (strikes[0], strikes[-1])

    # Stance: first strike to first off
    stance = (strikes[0], offs[0])

    # Swing: first off to second strike
    swing = (offs[0], strikes[1])

    return {
        "gait_cycle": gait_cycle,
        "stance": stance,
        "swing": swing,
    }



# Valid walking trial event sequences (7 events)
# Right side starts first
RIGHT_START_SEQUENCE = [
    ("Right", "Foot Strike"),
    ("Left", "Foot Off"),
    ("Left", "Foot Strike"),
    ("Right", "Foot Off"),
    ("Right", "Foot Strike"),
    ("Left", "Foot Off"),
    ("Left", "Foot Strike"),
]

# Left side starts first
LEFT_START_SEQUENCE = [
    ("Left", "Foot Strike"),
    ("Right", "Foot Off"),
    ("Right", "Foot Strike"),
    ("Left", "Foot Off"),
    ("Left", "Foot Strike"),
    ("Right", "Foot Off"),
    ("Right", "Foot Strike"),
]


def validate_walking_events(
    events_df: "pl.DataFrame",
    required_markers: list[str] | None = None,
) -> tuple[bool, str, list[dict]]:
    """Validate walking trial events from Parquet data.

    Checks:
    1. Exactly 7 events in correct alternating order
    2. Correct context/label pairs (Right/Left + Foot Strike/Foot Off)
    3. Required markers present (no gaps)

    Parameters
    ----------
    events_df : pl.DataFrame
        Events DataFrame with columns: time, context, label, frame,
        subject_id, session_id, trial_name.
    required_markers : list[str] or None
        If provided, check that these markers are present in the trial.

    Returns
    -------
    (is_valid, reason, violations)
        True if trial passes, reason string, list of violation details.
    """
    import polars as pl

    violations = []

    # Get events for this trial
    if events_df.is_empty():
        return False, "no events found", [{"type": "no_events"}]

    # Sort by time
    events_df = events_df.sort("time")

    # Check exactly 7 events
    n_events = len(events_df)
    if n_events != 7:
        violations.append({
            "type": "event_count",
            "expected": 7,
            "actual": n_events,
        })
        return False, f"expected 7 events, got {n_events}", violations

    # Extract context/label pairs
    event_sequence = [
        (row["context"], row["label"])
        for row in events_df.iter_rows(named=True)
    ]

    # Check against both valid sequences
    is_right_start = event_sequence == RIGHT_START_SEQUENCE
    is_left_start = event_sequence == LEFT_START_SEQUENCE

    if not is_right_start and not is_left_start:
        violations.append({
            "type": "event_order",
            "expected": "Right FS → Left FO → Left FS → Right FO → Right FS → Left FO → Left FS (or left-start variant)",
            "actual": " → ".join(f"{c} {l}" for c, l in event_sequence),
        })
        return False, "event sequence does not match expected pattern", violations

    return True, "valid", violations


def find_valid_walking_trials(
    markers_df: "pl.DataFrame",
    events_df: "pl.DataFrame",
    required_markers: list[str] | None = None,
) -> list[str]:
    """Find valid walking trials in a session.

    A trial is valid if:
    1. It has exactly 7 events in correct alternating order
    2. It is not a static trial
    3. Required markers are present (no gaps)

    Parameters
    ----------
    markers_df : pl.DataFrame
        Markers DataFrame for the session.
    events_df : pl.DataFrame
        Events DataFrame for the session.
    required_markers : list[str] or None
        Markers that must be present. If None, uses default set.

    Returns
    -------
    list of str
        List of valid trial names.
    """
    import polars as pl

    if required_markers is None:
        from .static_trial import REQUIRED_MARKERS
        required_markers = REQUIRED_MARKERS

    # Get unique trial names
    trial_names = markers_df["trial_name"].unique().to_list()

    valid_trials = []
    for trial_name in trial_names:
        # Skip static trials
        if "static" in trial_name.lower():
            continue

        # Get events for this trial
        trial_events = events_df.filter(pl.col("trial_name") == trial_name)

        # Validate events
        is_valid, reason, violations = validate_walking_events(trial_events)

        if not is_valid:
            logger.debug(f"  {trial_name}: {reason}")
            continue

        # Check required markers (only between first and last event)
        trial_markers = markers_df.filter(pl.col("trial_name") == trial_name)
        trial_events = events_df.filter(pl.col("trial_name") == trial_name)

        if not trial_events.is_empty():
            first_event_time = trial_events["time"].min()
            last_event_time = trial_events["time"].max()

            # Filter markers to the event window
            event_window_markers = trial_markers.filter(
                (pl.col("time") >= first_event_time) &
                (pl.col("time") <= last_event_time)
            )
            present_markers = set(event_window_markers["marker_name"].unique().to_list())
            missing = set(required_markers) - present_markers

            if missing:
                logger.debug(f"  {trial_name}: missing markers {missing}")
                continue

        valid_trials.append(trial_name)

    logger.info(f"Found {len(valid_trials)}/{len(trial_names)} valid walking trials")
    return valid_trials


def check_marker_gaps(
    marker_data: dict[str, np.ndarray],
    threshold: float = 0.0,
) -> tuple[bool, list[str]]:
    """Check for marker gaps (missing data) in a trial.

    Parameters
    ----------
    marker_data : dict
        Marker name -> (N, 3) array of marker positions.
    threshold : float
        Values at or below this threshold are considered missing.

    Returns
    -------
    (has_gaps, gap_markers)
        True if any marker has gaps, with list of marker names that have gaps.
    """
    gap_markers = []

    for name, data in marker_data.items():
        # A frame is "missing" if all coordinates are at or below threshold
        is_missing = np.all(np.abs(data) <= threshold, axis=1)
        if np.any(is_missing):
            gap_markers.append(name)

    return len(gap_markers) > 0, gap_markers
