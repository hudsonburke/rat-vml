"""Results storage — write IK/ID/Moco outputs to Parquet.

After each pipeline step (IK, ID, MocoInverse), the .sto/.mot output
is read and stored in Parquet for downstream analysis and plotting.

Usage::

    from rat_vml.analysis.results_storage import store_ik_result, store_id_result
    store_ik_result(ik_file, subject_id, session, trial_name, output_dir)
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _sto_to_long_df(filepath: Path, subject_id: str, session: str, trial_name: str) -> pl.DataFrame:
    """Read an OpenSim .sto/.mot file and convert to long Polars DataFrame.

    Returns DataFrame with columns: time, coord, value, subject_id, session_id, trial_name
    """
    from osimpy.io.sto import sto_to_df

    df, _ = sto_to_df(str(filepath))

    # Melt from wide to long format
    id_vars = ["time"]
    value_vars = [c for c in df.columns if c != "time"]

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, variable_name="coord", value_name="value")
    long_df = long_df.with_columns([
        pl.lit(subject_id).alias("subject_id"),
        pl.lit(session).alias("session_id"),
        pl.lit(trial_name).alias("trial_name"),
    ])

    return long_df


def store_ik_result(
    ik_file: Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: Path,
) -> Path:
    """Store IK result in Parquet.

    Parameters
    ----------
    ik_file : Path
        Path to IK .mot file.
    subject_id, session, trial_name : str
        Identifiers for this trial.
    output_dir : Path
        Directory to write ik_results.parquet.

    Returns
    -------
    Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _sto_to_long_df(ik_file, subject_id, session, trial_name)
    output_path = output_dir / "ik_results.parquet"

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        df = pl.concat([existing, df])

    df.write_parquet(output_path)
    logger.info(f"  Stored IK result: {len(df)} rows -> {output_path}")
    return output_path


def store_id_result(
    id_file: Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: Path,
) -> Path:
    """Store ID result in Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _sto_to_long_df(id_file, subject_id, session, trial_name)
    output_path = output_dir / "id_results.parquet"

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        df = pl.concat([existing, df])

    df.write_parquet(output_path)
    logger.info(f"  Stored ID result: {len(df)} rows -> {output_path}")
    return output_path


def store_moco_result(
    moco_file: Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: Path,
) -> Path:
    """Store MocoInverse result in Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _sto_to_long_df(moco_file, subject_id, session, trial_name)
    output_path = output_dir / "moco_results.parquet"

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        df = pl.concat([existing, df])

    df.write_parquet(output_path)
    logger.info(f"  Stored Moco result: {len(df)} rows -> {output_path}")
    return output_path
