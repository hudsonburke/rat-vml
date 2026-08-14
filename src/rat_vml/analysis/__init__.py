"""Rat VML analysis workflows.

Composes movedb-core's catalog and adapters with rat-specific
analysis pipelines for Inverse Kinematics, Inverse Dynamics,
MocoInverse muscle analysis, and result plotting.

Modules
-------
parquet_io      : Parquet→TRC and Parquet→MOT export for OpenSim
events          : Gait event data structures and trial validation
plots           : Manuscript-quality kinematic and kinetic figures
filtering       : Marker and force plate filtering (Butterworth, notch)
results_storage : Write IK/ID/Moco outputs to Parquet
aggregation     : Group aggregation and SPM t-tests
subject_groups  : Subject-to-treatment-group mapping
static_trial    : Static trial selection for scaling
"""
