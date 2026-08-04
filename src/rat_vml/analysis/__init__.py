"""Rat hindlimb analysis workflows.

Composes osimpy's generic OpenSim tool wrappers into rat-specific
analysis pipelines for Inverse Kinematics, Inverse Dynamics, MocoInverse
muscle analysis, and result plotting.

Modules
-------
io         : C3D→TRC and C3D→FP MOT export (Vicon→OpenSim transform, filtering)
events     : Trial validation, gait event handling, marker gap detection
forces     : Force plate processing (Vicon→OpenSim transform, filtering, MOT export)
pipeline   : End-to-end analysis pipeline (scale, IK, ID, MocoInverse, spline, group aggregation)
plots      : Manuscript-quality kinematic and kinetic figures
queries    : DuckDB query templates for the Rerun .rrd catalog
defaults   : Rat-specific constants (coordinate names, marker sets)
subject_groups : Subject-to-treatment-group mapping from AFIRM spreadsheet
"""

from .pipeline import run_ik, run_id, run_moco, run_subject

__all__ = ["run_ik", "run_id", "run_moco", "run_subject"]
