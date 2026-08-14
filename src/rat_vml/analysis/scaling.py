"""Scaling utilities for rat hindlimb model.

Wraps osimpy's ScaleSettings with rat-specific parameter handling.
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
    marker_path: Path,
    output_dir: Path,
) -> Path:
    """Scale base model to subject anthropometrics.

    Parameters
    ----------
    base_model : Path
        Path to unscaled model (.osim).
    setup_path : Path
        Path to scale setup XML template.
    marker_set_path : Path
        Path to marker set XML file.
    subject_name : str
        Subject identifier for output file naming.
    mass : float
        Subject mass (kg).
    marker_path : Path
        Path to TRC file with marker positions.
    output_dir : Path
        Output directory for scaled model.

    Returns
    -------
    Path
        Path to scaled model.
    """
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    if not setup_path.exists():
        raise FileNotFoundError(f"Scale setup not found: {setup_path}")
    if not marker_set_path.exists():
        raise FileNotFoundError(f"Marker set not found: {marker_set_path}")
    if not marker_path.exists():
        raise FileNotFoundError(f"Marker file not found: {marker_path}")

    output_path = (output_dir / f"{subject_name}_scaled.osim").resolve()

    from osimpy.tools import ScaleSettings

    settings = ScaleSettings(
        name=f"{subject_name}_scale",
        setup_path=setup_path,
        model_path=base_model,
        results_directory=output_dir.resolve(),
        marker_set_path=marker_set_path,
        marker_path=marker_path,
        output_model_file=str(output_path),
        subject_mass=mass,
        preserve_mass_distribution=False,
    )

    result = settings.run()

    if result.success:
        logger.info(f"Scaled model: {output_path}")
        return output_path
    else:
        logger.error(f"Scaling failed: {result.errors}")
        raise RuntimeError(f"Scaling failed: {result.errors}")
