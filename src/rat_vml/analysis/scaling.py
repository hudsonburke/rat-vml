"""Scaling utilities for rat hindlimb model.

Thin wrapper around rathindlimb.scale.scale_opensim_model.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def scale_model_for_subject(
    base_model: Path,
    setup_path: Path,
    marker_set_path: Path,
    subject_name: str,
    mass: float,
    r_femur_length: float,
    r_tibia_length: float,
    r_foot_length: float,
    l_femur_length: float | None = None,
    l_tibia_length: float | None = None,
    l_foot_length: float | None = None,
    marker_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Scale base model to subject anthropometrics.

    Uses rathindlimb.scale.scale_opensim_model which runs OpenSim Scale Tool
    and applies Hicks regression for masses, COM, and inertias.
    """
    from rathindlimb.scale import scale_opensim_model, RatScalingParameters

    if output_dir is None:
        output_dir = base_model.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use right side values as defaults for left
    if l_femur_length is None:
        l_femur_length = r_femur_length
    if l_tibia_length is None:
        l_tibia_length = r_tibia_length
    if l_foot_length is None:
        l_foot_length = r_foot_length

    parameters: RatScalingParameters = {
        "Mass": mass,
        "RFemurLength": r_femur_length,
        "RTibiaLength": r_tibia_length,
        "RFootLength": r_foot_length,
        "LFemurLength": l_femur_length,
        "LTibiaLength": l_tibia_length,
        "LFootLength": l_foot_length,
    }

    scale_opensim_model(
        name=subject_name,
        trc_file_name=str(marker_path) if marker_path else "",
        parameters=parameters,
        output_dir=str(output_dir),
    )

    scaled_path = output_dir / f"{subject_name}_scaled.osim"
    logger.info(f"Scaled model: {scaled_path}")
    return scaled_path
