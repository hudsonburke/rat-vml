from movedb.osim import export_trc, export_external_loads
from movedb.ingest import C3DAdapter
import numpy as np
import polars as pl


trial_table = pl.read_csv("data/trials.csv")

# Get distinct Subject + Session combinations
session_files = {}
# Process each trial's mocap data (c3d and vicon files)
for sess, trials in session_files.items():
    for trial in trials:
        c3d = C3DAdapter.from_file(trial)
        t = c3d.to_trial()
        marker_df = t.markers_to_dataframe()
        export_trc()

        # For the static trial create the scaled model
        # For each dynamic trial run IK and ID using the scaled model
        export_external_loads()

        