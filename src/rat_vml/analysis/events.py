"""Gait event data structures and validation.

Provides the GaitEvents dataclass and validation logic for walking trials.
Event extraction from C3D/ENF files has been removed — events now come
from Parquet files written by movedb-core.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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
