"""Rat VML Data Explorer — Interactive notebook for browsing the dataset.

Usage:
    marimo edit notebooks/explore.py

This notebook provides interactive exploration of the rat hindlimb
motion capture data stored as Parquet files.
"""

import marimo

__generated_with = "0.0.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    from movedb import MoveDB
    return mo, pl, MoveDB, Path


@app.cell
def _(mo, MoveDB, Path):
    mo.md("# Rat VML Data Explorer")
    mo.md("Browse subjects, sessions, trials, and view marker/force plate data.")
    return


@app.cell
def _(mo, MoveDB, Path):
    data_dir = Path("data/processed")
    if not data_dir.exists():
        mo.md("⚠️ Data directory not found. Run `snapshot_download` first.")
        mo.stop()

    db = MoveDB(data_dir)
    subjects = db.subjects()
    return db, subjects


@app.cell
def _(mo, subjects):
    subject_select = mo.ui.dropdown(
        options=subjects,
        label="Subject",
        value=subjects[0] if subjects else None,
    )
    subject_select
    return (subject_select,)


@app.cell
def _(mo, db, subject_select):
    sessions = db.sessions(subject_select.value) if subject_select.value else []
    session_select = mo.ui.dropdown(
        options=sessions,
        label="Session",
        value=sessions[0] if sessions else None,
    )
    session_select
    return (session_select,)


@app.cell
def _(mo, db, subject_select, session_select):
    trials = db.trials(subject_select.value, session_select.value) if session_select.value else []
    trial_select = mo.ui.dropdown(
        options=trials,
        label="Trial",
        value=trials[0] if trials else None,
    )
    trial_select
    return (trial_select,)


@app.cell
def _(mo, db, subject_select, session_select):
    mo.md("## Session Parameters")
    params = db.get_parameters(subject_select.value, session=session_select.value)
    if not params.is_empty():
        # Show key parameters
        key_cols = [c for c in ["Mass", "RFemurLength", "RTibiaLength", "RFootLength",
                                 "LFemurLength", "LTibiaLength", "LFootLength"]
                    if c in params.columns]
        if key_cols:
            mo.ui.table(params.select(key_cols))
        else:
            mo.md("No key parameters found")
    else:
        mo.md("No parameters available")
    return


@app.cell
def _(mo, db, subject_select, session_select, trial_select):
    mo.md("## Gait Events")
    if trial_select.value:
        events = db.get_events(subject_select.value, session=session_select.value)
        trial_events = events.filter(pl.col("trial_name") == trial_select.value)
        if not trial_events.is_empty():
            mo.ui.table(trial_events)
        else:
            mo.md("No events for this trial")
    else:
        mo.md("Select a trial")
    return


@app.cell
def _(mo, db, subject_select, session_select, trial_select):
    mo.md("## Marker Trajectories")
    if trial_select.value:
        markers = db.get_points(subject_select.value, session=session_select.value)
        trial_markers = markers.filter(pl.col("trial_name") == trial_select.value)

        # Get unique markers
        marker_names = trial_markers["marker_name"].unique().to_list()

        marker_select = mo.ui.multiselect(
            options=marker_names,
            value=marker_names[:3] if len(marker_names) >= 3 else marker_names,
            label="Markers to plot",
        )
        marker_select

        if marker_select.value:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for marker in marker_select.value:
                m = trial_markers.filter(pl.col("marker_name") == marker)
                axes[0].plot(m["time"], m["x"], label=marker)
                axes[1].plot(m["time"], m["y"], label=marker)
                axes[2].plot(m["time"], m["z"], label=marker)

            axes[0].set_ylabel("X (mm)")
            axes[1].set_ylabel("Y (mm)")
            axes[2].set_ylabel("Z (mm)")
            axes[2].set_xlabel("Time (s)")
            for ax in axes:
                ax.legend(fontsize=8)
            plt.tight_layout()
            plt.gcf()
        else:
            mo.md("Select markers to plot")
    else:
        mo.md("Select a trial")
    return


@app.cell
def _(mo, db, subject_select, session_select, trial_select):
    mo.md("## Force Plate Data")
    if trial_select.value:
        fps = db.get_forceplates(subject_select.value, session=session_select.value)
        trial_fps = fps.filter(pl.col("trial_name") == trial_select.value)

        if not trial_fps.is_empty():
            # Filter to force z (vertical force)
            force_z = trial_fps.filter(
                (pl.col("variable") == "force") & (pl.col("axis") == "z")
            )

            if not force_z.is_empty():
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 4))
                for fp in force_z["fp_name"].unique().to_list():
                    fp_data = force_z.filter(pl.col("fp_name") == fp)
                    ax.plot(fp_data["time"], fp_data["value"], label=fp)
                ax.set_ylabel("Force Z (N)")
                ax.set_xlabel("Time (s)")
                ax.legend()
                plt.tight_layout()
                plt.gcf()
            else:
                mo.md("No vertical force data")
        else:
            mo.md("No force plate data for this trial")
    else:
        mo.md("Select a trial")
    return


@app.cell
def _(mo, db, subject_select, session_select):
    mo.md("## Trial Summary")
    if session_select.value:
        trials = db.trials(subject_select.value, session_select.value)
        events = db.get_events(subject_select.value, session=session_select.value)

        # Count events per trial
        if not events.is_empty():
            trial_counts = (
                events.filter(pl.col("session_id") == session_select.value)
                .group_by("trial_name")
                .agg(pl.len().alias("n_events"))
                .sort("trial_name")
            )

            # Mark valid trials (7+ events)
            trial_counts = trial_counts.with_columns(
                pl.when(pl.col("n_events") >= 7)
                .then(pl.lit("✓"))
                .otherwise(pl.lit("✗"))
                .alias("valid")
            )

            mo.ui.table(trial_counts)
        else:
            mo.md("No events available")
    else:
        mo.md("Select a session")
    return
