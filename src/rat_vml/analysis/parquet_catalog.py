"""Parquet-based catalog queries for rat-vml.

Queries Parquet files written by movedb-core's ingestion module.
Replaces the .rrd-based RerunCatalog with direct Parquet reads.

Usage::

    from rat_vml.analysis.parquet_catalog import ParquetCatalog

    cat = ParquetCatalog("data/processed/")
    subjects = cat.subjects_by_group("No Repair")
    valid = cat.valid_walking_trials(min_events=7)
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class ParquetCatalog:
    """Query interface for Parquet files written by movedb-core.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing subject subdirectories with Parquet files.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _read_parquet(self, subject_id: str, name: str) -> pl.DataFrame | None:
        """Read a Parquet file for a subject."""
        path = self.data_dir / subject_id / name
        if path.exists():
            return pl.read_parquet(path)
        return None

    # ------------------------------------------------------------------
    # Subject-level queries
    # ------------------------------------------------------------------

    def all_subjects(self) -> pl.DataFrame:
        """List all subjects with their group and session metadata.

        Returns DataFrame with columns: subject_id, group, session_id
        """
        markers = self._read_parquet_all("markers.parquet")
        if markers is None or markers.is_empty():
            return pl.DataFrame(schema={"subject_id": pl.Utf8, "group": pl.Utf8, "session_id": pl.Utf8})

        # Get unique subject/session combinations
        subjects = markers.select(["subject_id", "session_id"]).unique()

        # Read events to get group info if available
        events = self._read_parquet_all("events.parquet")
        if events is not None and "group" in events.columns:
            groups = events.select(["subject_id", "group"]).unique()
            subjects = subjects.join(groups, on="subject_id", how="left")

        return subjects.sort("subject_id")

    def subjects_by_group(self, group: str) -> pl.DataFrame:
        """List subjects belonging to a specific treatment group.

        Parameters
        ----------
        group : str
            Treatment group name (e.g. "No Repair", "TEMR", "Control").
        """
        markers = self._read_parquet_all("markers.parquet")
        if markers is None or markers.is_empty():
            return pl.DataFrame(schema={"subject_id": pl.Utf8})

        # Get unique subjects
        subjects = markers.select("subject_id").unique()

        # Filter by group if events have group info
        events = self._read_parquet_all("events.parquet")
        if events is not None and "group" in events.columns:
            group_subjects = events.filter(pl.col("group") == group).select("subject_id").unique()
            subjects = subjects.join(group_subjects, on="subject_id", how="inner")

        return subjects.sort("subject_id")

    # ------------------------------------------------------------------
    # Trial-level queries
    # ------------------------------------------------------------------

    def all_trials(self) -> pl.DataFrame:
        """List all trials with subject, session, and trial name.

        Returns DataFrame with columns: subject_id, session_id, trial_name
        """
        markers = self._read_parquet_all("markers.parquet")
        if markers is None or markers.is_empty():
            return pl.DataFrame(schema={"subject_id": pl.Utf8, "session_id": pl.Utf8, "trial_name": pl.Utf8})

        return markers.select(["subject_id", "session_id", "trial_name"]).unique().sort(["subject_id", "session_id", "trial_name"])

    def trials_by_session(self, session: str) -> pl.DataFrame:
        """List all trials for a specific session.

        Parameters
        ----------
        session : str
            Session name (e.g. "Baseline", "Week24").
        """
        markers = self._read_parquet_all("markers.parquet")
        if markers is None or markers.is_empty():
            return pl.DataFrame(schema={"subject_id": pl.Utf8, "trial_name": pl.Utf8})

        return (
            markers.filter(pl.col("session_id") == session)
            .select(["subject_id", "trial_name"])
            .unique()
            .sort(["subject_id", "trial_name"])
        )

    # ------------------------------------------------------------------
    # Event-based trial filtering
    # ------------------------------------------------------------------

    def trial_event_counts(self) -> pl.DataFrame:
        """Count events per trial.

        Returns DataFrame with columns: subject_id, session_id, trial_name, n_events
        """
        events = self._read_parquet_all("events.parquet")
        if events is None or events.is_empty():
            return pl.DataFrame(schema={
                "subject_id": pl.Utf8, "session_id": pl.Utf8,
                "trial_name": pl.Utf8, "n_events": pl.Int64,
            })

        return (
            events.group_by(["subject_id", "session_id", "trial_name"])
            .agg(pl.col("label").count().alias("n_events"))
            .sort(["subject_id", "session_id", "trial_name"])
        )

    def valid_walking_trials(
        self,
        min_events: int = 7,
        session: str | None = None,
        group: str | None = None,
    ) -> pl.DataFrame:
        """Find walking trials with sufficient gait events.

        Parameters
        ----------
        min_events : int
            Minimum number of events required (default 7).
        session : str or None
            Filter to a specific session (e.g. "Week24").
        group : str or None
            Filter to a specific treatment group.
        """
        events = self._read_parquet_all("events.parquet")
        if events is None or events.is_empty():
            return pl.DataFrame(schema={
                "subject_id": pl.Utf8, "session_id": pl.Utf8,
                "trial_name": pl.Utf8, "n_events": pl.Int64,
            })

        # Filter by session
        if session:
            events = events.filter(pl.col("session_id") == session)

        # Count events per trial
        trial_counts = (
            events.group_by(["subject_id", "session_id", "trial_name"])
            .agg(pl.col("label").count().alias("n_events"))
            .filter(pl.col("n_events") >= min_events)
        )

        # Filter by group
        if group and "group" in events.columns:
            group_subjects = events.filter(pl.col("group") == group).select("subject_id").unique()
            trial_counts = trial_counts.join(group_subjects, on="subject_id", how="inner")

        return trial_counts.sort(["subject_id", "session_id", "trial_name"])

    # ------------------------------------------------------------------
    # IK/ID result queries
    # ------------------------------------------------------------------

    def group_ik_data(
        self,
        group: str,
        session: str,
        coord_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """Get IK data for all trials in a group/session.

        Returns DataFrame with columns: subject_id, trial_name, coord, value
        """
        ik = self._read_parquet_all("ik_results.parquet")
        if ik is None or ik.is_empty():
            return pl.DataFrame(schema={
                "subject_id": pl.Utf8, "trial_name": pl.Utf8,
                "coord": pl.Utf8, "value": pl.Float64,
            })

        # Filter by session
        ik = ik.filter(pl.col("session_id") == session)

        # Filter by group
        events = self._read_parquet_all("events.parquet")
        if events is not None and "group" in events.columns:
            group_subjects = events.filter(pl.col("group") == group).select("subject_id").unique()
            ik = ik.join(group_subjects, on="subject_id", how="inner")

        # Filter by coord names
        if coord_names:
            ik = ik.filter(pl.col("coord").is_in(coord_names))

        return ik.select(["subject_id", "trial_name", "coord", "value"]).sort(["subject_id", "trial_name", "coord"])

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def group_summary(self, session: str) -> pl.DataFrame:
        """Summary of subjects, trials, and event counts per group."""
        events = self._read_parquet_all("events.parquet")
        if events is None or events.is_empty():
            return pl.DataFrame(schema={
                "group": pl.Utf8, "n_subjects": pl.Int64,
                "n_trials": pl.Int64, "mean_events": pl.Float64,
                "valid_trials": pl.Int64,
            })

        # Filter by session
        events = events.filter(pl.col("session_id") == session)

        # Count events per trial
        trial_events = (
            events.group_by(["subject_id", "trial_name"])
            .agg(pl.col("label").count().alias("n_events"))
        )

        # Get group info
        if "group" in events.columns:
            subject_groups = events.select(["subject_id", "group"]).unique()
            trial_events = trial_events.join(subject_groups, on="subject_id", how="left")

            # Aggregate by group
            summary = (
                trial_events.group_by("group")
                .agg([
                    pl.col("subject_id").n_unique().alias("n_subjects"),
                    pl.col("trial_name").count().alias("n_trials"),
                    pl.col("n_events").mean().round(1).alias("mean_events"),
                    (pl.col("n_events") >= 7).sum().alias("valid_trials"),
                ])
                .sort("group")
            )
        else:
            summary = pl.DataFrame(schema={
                "group": pl.Utf8, "n_subjects": pl.Int64,
                "n_trials": pl.Int64, "mean_events": pl.Float64,
                "valid_trials": pl.Int64,
            })

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_parquet_all(self, name: str) -> pl.DataFrame | None:
        """Read and concatenate a Parquet file from all subjects."""
        dfs = []
        for subject_dir in sorted(self.data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            df = self._read_parquet(subject_dir.name, name)
            if df is not None and not df.is_empty():
                dfs.append(df)

        if not dfs:
            return None

        return pl.concat(dfs)
