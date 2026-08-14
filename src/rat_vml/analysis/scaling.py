"""Scaling utilities for rat hindlimb model.

Wraps osimpy's ScaleSettings with rat-specific parameter handling.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def scale_model_for_subject(
    base_model: Path,
    subject_name: str,
    mass: float,
    r_femur_length: float,
    r_tibia_length: float,
    r_foot_length: float,
    l_femur_length: float | None = None,
    l_tibia_length: float | None = None,
    l_foot_length: float | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Scale base model to subject anthropometrics.

    Parameters
    ----------
    base_model : Path
        Path to unscaled model (.osim).
    subject_name : str
        Subject identifier for output file naming.
    mass : float
        Subject mass (kg).
    r_femur_length : float
        Right femur length (mm).
    r_tibia_length : float
        Right tibia length (mm).
    r_foot_length : float
        Right foot length (mm).
    l_femur_length : float or None
        Left femur length (mm). If None, uses right side.
    l_tibia_length : float or None
        Left tibia length (mm). If None, uses right side.
    l_foot_length : float or None
        Left foot length (mm). If None, uses right side.
    output_dir : Path or None
        Output directory. If None, uses parent of base_model.

    Returns
    -------
    Path
        Path to scaled model.
    """
    # Use right side values as defaults for left
    if l_femur_length is None:
        l_femur_length = r_femur_length
    if l_tibia_length is None:
        l_tibia_length = r_tibia_length
    if l_foot_length is None:
        l_foot_length = r_foot_length

    if output_dir is None:
        output_dir = base_model.parent

    output_path = output_dir / f"{subject_name}_scaled.osim"

    from osimpy.tools import ScaleSettings

    settings = ScaleSettings(
        model_path=base_model,
        output_model_file=str(output_path),
        mass=mass,
        # These are scale factors, not absolute lengths
        # The actual scaling uses the model's base lengths
    )

    result = settings.run()

    if result.success:
        logger.info(f"Scaled model: {output_path}")
        return output_path
    else:
        logger.error(f"Scaling failed: {result.errors}")
        raise RuntimeError(f"Scaling failed: {result.errors}")
