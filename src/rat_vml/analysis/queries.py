"""DuckDB query templates for the rat-vml MoveDB catalog.

These queries run against .rrd files using the Rerun DuckDB extension.
Entity paths follow the MoveDB convention:

    {subject}/subject/body_measurements/{param}   — Static subject params
    {subject}/subject/metadata/group               — Treatment group
    {subject}/subject/metadata/session             — Session
    {subject}/trials/{session}_{trial}/markers     — Per-frame markers
    {subject}/trials/{session}_{trial}/events/{label} — Trial events
    {subject}/trials/{session}_{trial}/ik/{dof}    — IK joint angles
    {subject}/trials/{session}_{trial}/id/{dof}    — ID joint moments

Usage::

    from rat_vml.analysis.queries import RerunCatalog

    cat = RerunCatalog("data/rrd/")
    subjects = cat.subjects_by_group("No Repair")
    valid = cat.valid_walking_trials(min_events=7)
    ik = cat.group_ik_data("Control", "Week24")
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class RerunCatalog:
    """Query interface for a Rerun .rrd catalog via DuckDB.

    Parameters
    ----------
    rrd_dir : str or Path
        Directory containing .rrd files (one per subject).
    """

    def __init__(self, rrd_dir: str | Path):
        self.rrd_dir = str(Path(rrd_dir).resolve())
        self._conn = None

    def _connect(self):
        """Lazy connection — only opens DuckDB when first query is made."""
        if self._conn is not None:
            return self._conn

        import duckdb
        self._conn = duckdb.connect()
        try:
            self._conn.execute("INSTALL rrd FROM community;")
            self._conn.execute("LOAD rrd;")
        except Exception:
            self._conn.execute("LOAD rrd;")
        self._conn.execute(f"CALL rrd_scan_directory('{self.rrd_dir}', 'bio');")
        return self._conn

    def _query(self, sql: str) -> pl.DataFrame:
        """Execute SQL and return results as a Polars DataFrame."""
        conn = self._connect()
        result = conn.execute(sql)
        rows = result.fetchall()
        cols = [d[0] for d in result.description]
        return pl.DataFrame(rows, schema=cols, orient="row")

    # ------------------------------------------------------------------
    # Subject-level queries
    # ------------------------------------------------------------------

    def all_subjects(self) -> pl.DataFrame:
        """List all subjects with their group and session metadata.

        Returns DataFrame with columns: subject, group, session
        """
        return self._query("""
            SELECT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                MAX(CASE WHEN entity_path LIKE '%/metadata/group'
                    THEN value END) AS "group",
                MAX(CASE WHEN entity_path LIKE '%/metadata/session'
                    THEN value END) AS session
            FROM bio
            WHERE entity_path LIKE '%/subject/metadata/%'
            GROUP BY SPLIT_PART(entity_path, '/', 2)
            ORDER BY subject
        """)

    def subjects_by_group(self, group: str) -> pl.DataFrame:
        """List subjects belonging to a specific treatment group.

        Parameters
        ----------
        group : str
            Treatment group name (e.g. "No Repair", "TEMR", "Control").
        """
        return self._query(f"""
            SELECT DISTINCT SPLIT_PART(entity_path, '/', 2) AS subject
            FROM bio
            WHERE entity_path LIKE '%/metadata/group'
            AND value = '{group}'
            ORDER BY subject
        """)

    def body_measurements(self, subject: str | None = None) -> pl.DataFrame:
        """Get body measurements for one or all subjects.

        Returns DataFrame with columns: subject, param, value
        """
        where = ""
        if subject:
            where = f"AND SPLIT_PART(entity_path, '/', 2) = '{subject}'"
        return self._query(f"""
            SELECT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                SPLIT_PART(entity_path, '/', -1) AS param,
                value
            FROM bio
            WHERE entity_path LIKE '%/body_measurements/%' {where}
            ORDER BY subject, param
        """)

    # ------------------------------------------------------------------
    # Trial-level queries
    # ------------------------------------------------------------------

    def all_trials(self) -> pl.DataFrame:
        """List all trials with subject, session, and trial name.

        Returns DataFrame with columns: subject, session, trial, entity_prefix
        """
        return self._query("""
            SELECT DISTINCT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1) AS session,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2) AS trial,
                SPLIT_PART(entity_path, '/', 2) || '/trials/' ||
                    SPLIT_PART(entity_path, '/', 4) AS entity_prefix
            FROM bio
            WHERE entity_path LIKE '%/trials/%/markers'
            ORDER BY subject, session, trial
        """)

    def trials_by_session(self, session: str) -> pl.DataFrame:
        """List all trials for a specific session.

        Parameters
        ----------
        session : str
            Session name (e.g. "Baseline", "Week24").
        """
        return self._query(f"""
            SELECT DISTINCT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2) AS trial,
                SPLIT_PART(entity_path, '/', 2) || '/trials/' ||
                    SPLIT_PART(entity_path, '/', 4) AS entity_prefix
            FROM bio
            WHERE entity_path LIKE '%/trials/{session}_%/markers'
            ORDER BY subject, trial
        """)

    # ------------------------------------------------------------------
    # Event-based trial filtering
    # ------------------------------------------------------------------

    def trial_event_counts(self) -> pl.DataFrame:
        """Count events per trial.

        Returns DataFrame with columns: subject, session, trial, n_events
        """
        return self._query("""
            SELECT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1) AS session,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2) AS trial,
                COUNT(*) AS n_events
            FROM bio
            WHERE entity_path LIKE '%/trials/%/events/%'
            GROUP BY
                SPLIT_PART(entity_path, '/', 2),
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1),
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2)
            ORDER BY subject, session, trial
        """)

    def valid_walking_trials(
        self,
        min_events: int = 7,
        session: str | None = None,
        group: str | None = None,
    ) -> pl.DataFrame:
        """Find walking trials with sufficient gait events.

        A trial is considered valid if it has at least `min_events` gait
        events (foot strikes + foot offs).

        Parameters
        ----------
        min_events : int
            Minimum number of events required (default 7).
        session : str or None
            Filter to a specific session (e.g. "Week24").
        group : str or None
            Filter to a specific treatment group.
        """
        session_filter = f"AND SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1) = '{session}'" if session else ""

        # Get trials with sufficient events
        trials = self._query(f"""
            SELECT
                SPLIT_PART(entity_path, '/', 2) AS subject,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1) AS session,
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2) AS trial,
                COUNT(*) AS n_events
            FROM bio
            WHERE entity_path LIKE '%/trials/%/events/%' {session_filter}
            GROUP BY
                SPLIT_PART(entity_path, '/', 2),
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 1),
                SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2)
            HAVING COUNT(*) >= {min_events}
            ORDER BY subject, session, trial
        """)

        if group and not trials.is_empty():
            # Filter by group
            group_subjects = self.subjects_by_group(group)
            if not group_subjects.is_empty():
                subjects = group_subjects["subject"].to_list()
                trials = trials.filter(pl.col("subject").is_in(subjects))

        return trials

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

        Returns DataFrame with columns: subject, trial, coord, value
        (one row per frame per coordinate per trial).
        """
        coord_filter = ""
        if coord_names:
            coords = ", ".join(f"'{c}'" for c in coord_names)
            coord_filter = f"AND SPLIT_PART(entity_path, '/', -1) IN ({coords})"

        return self._query(f"""
            SELECT
                t.subject,
                SPLIT_PART(SPLIT_PART(t.entity_path, '/', 4), '_', 2) AS trial,
                SPLIT_PART(t.entity_path, '/', -1) AS coord,
                t.value
            FROM bio t
            JOIN bio g ON t.subject = g.subject
            WHERE t.entity_path LIKE '%/trials/{session}_%/ik/%'
            AND g.entity_path LIKE '%/metadata/group'
            AND g.value = '{group}'
            {coord_filter}
            ORDER BY t.subject, trial, coord
        """)

    def group_id_data(
        self,
        group: str,
        session: str,
        moment_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """Get ID data for all trials in a group/session.

        Returns DataFrame with columns: subject, trial, moment, value
        """
        moment_filter = ""
        if moment_names:
            moments = ", ".join(f"'{m}'" for m in moment_names)
            moment_filter = f"AND SPLIT_PART(entity_path, '/', -1) IN ({moments})"

        return self._query(f"""
            SELECT
                t.subject,
                SPLIT_PART(SPLIT_PART(t.entity_path, '/', 4), '_', 2) AS trial,
                SPLIT_PART(t.entity_path, '/', -1) AS moment,
                t.value
            FROM bio t
            JOIN bio g ON t.subject = g.subject
            WHERE t.entity_path LIKE '%/trials/{session}_%/id/%'
            AND g.entity_path LIKE '%/metadata/group'
            AND g.value = '{group}'
            {moment_filter}
            ORDER BY t.subject, trial, moment
        """)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def group_summary(self, session: str) -> pl.DataFrame:
        """Summary of subjects, trials, and event counts per group.

        Returns DataFrame with columns: group, n_subjects, n_trials,
        mean_events, valid_trials (≥7 events)
        """
        return self._query(f"""
            WITH trial_events AS (
                SELECT
                    SPLIT_PART(entity_path, '/', 2) AS subject,
                    SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2) AS trial,
                    COUNT(*) AS n_events
                FROM bio
                WHERE entity_path LIKE '%/trials/{session}_%/events/%'
                GROUP BY
                    SPLIT_PART(entity_path, '/', 2),
                    SPLIT_PART(SPLIT_PART(entity_path, '/', 4), '_', 2)
            ),
            subject_groups AS (
                SELECT
                    SPLIT_PART(entity_path, '/', 2) AS subject,
                    value AS "group"
                FROM bio
                WHERE entity_path LIKE '%/metadata/group'
            )
            SELECT
                sg."group",
                COUNT(DISTINCT te.subject) AS n_subjects,
                COUNT(*) AS n_trials,
                ROUND(AVG(te.n_events), 1) AS mean_events,
                SUM(CASE WHEN te.n_events >= 7 THEN 1 ELSE 0 END) AS valid_trials
            FROM trial_events te
            JOIN subject_groups sg ON te.subject = sg.subject
            GROUP BY sg."group"
            ORDER BY sg."group"
        """)

    def close(self):
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
