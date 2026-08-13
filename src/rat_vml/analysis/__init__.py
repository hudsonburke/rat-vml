"""Rat hindlimb analysis workflows.

Composes osimpy's generic OpenSim tool wrappers into rat-specific
analysis pipelines for Inverse Kinematics, Inverse Dynamics, MocoInverse
muscle analysis, and result plotting.

Modules
-------nparquet_io   : Parquet→TRC and Parquet→MOT export for OpenSim
events     : Gait event data structures and trial validation
pipeline   : End-to-end analysis pipeline (scale, IK, ID, MocoInverse, group aggregation)
plots      : Manuscript-quality kinematic and kinetic figures
filtering  : Marker and force plate filtering (Butterworth, notch)
parquet_catalog : Parquet-based catalog queries
results_storage : Write IK/ID/Moco outputs to Parquet
aggregation : Group aggregation and SPM t-tests
defaults   : Rat-specific constants (coordinate names, marker sets)
subject_groups : Subject-to-treatment-group mapping from AFIRM spreadsheet
static_trial : Static trial selection for scaling
"""

from .pipeline import run_ik, run_id, run_moco, run_subject

__all__ = ["run_ik", "run_id", "run_moco", "run_subject"]
