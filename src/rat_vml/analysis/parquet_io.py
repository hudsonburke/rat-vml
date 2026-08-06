"""Parquet -> OpenSim file format extraction.

Reads marker and force plate data from Parquet files (written by
movedb-core) and writes TRC and MOT files for OpenSim IK/ID.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_VICON_TO_OPENSIM = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)


def _read_markers(data_dir: Path, subject_id: str, session: str, trial_name: str) -> pl.DataFrame:
    path = data_dir / subject_id / "markers.parquet"
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(path)
    return df.filter(
        (pl.col("subject_id") == subject_id)
        & (pl.col("session_id") == session)
        & (pl.col("trial_name") == trial_name)
    )


def _read_forceplates(data_dir: Path, subject_id: str, session: str, trial_name: str) -> pl.DataFrame:
    path = data_dir / subject_id / "forceplates.parquet"
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(path)
    return df.filter(
        (pl.col("subject_id") == subject_id)
        & (pl.col("session_id") == session)
        & (pl.col("trial_name") == trial_name)
    )


def _detect_rate(df: pl.DataFrame) -> float:
    times = df["time"].unique().sort().to_list()
    if len(times) < 2:
        return 200.0
    dt = times[1] - times[0]
    return round(1.0 / dt, 1) if dt > 0 else 200.0


def parquet_to_trc(
    data_dir: str | Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: str | Path,
    output_name: str | None = None,
) -> Path:
    """Extract markers from Parquet and write TRC file."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = f"{trial_name}.trc"

    markers_df = _read_markers(data_dir, subject_id, session, trial_name)
    if markers_df.is_empty():
        raise ValueError(f"No marker data for {subject_id}/{session}/{trial_name}")

    marker_names = sorted(markers_df["marker_name"].unique().to_list())
    rate = _detect_rate(markers_df)
    frames = markers_df.select(["time", "frame"]).unique().sort("frame")
    n_frames = len(frames)

    header_rows = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_name}",
        f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate}\t{rate}\t{n_frames}\t{len(marker_names)}\tmm\t{rate}\t1\t{n_frames}",
        "Frame#\tTime\t" + "\t".join(f"{name}\t\t" for name in marker_names),
        "\t\t" + "\t".join("X\tY\tZ" for _ in marker_names),
    ]

    data_rows = []
    for frame_row in frames.iter_rows(named=True):
        frame = frame_row["frame"]
        time = frame_row["time"]
        frame_markers = markers_df.filter(pl.col("frame") == frame)

        coords = []
        for name in marker_names:
            marker = frame_markers.filter(pl.col("marker_name") == name)
            if marker.is_empty():
                coords.extend(["0", "0", "0"])
            else:
                x = float(marker["x"].to_list()[0])
                y = float(marker["y"].to_list()[0])
                z = float(marker["z"].to_list()[0])
                v = np.array([x, y, z]) @ _VICON_TO_OPENSIM.T
                coords.extend([f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}"])

        data_rows.append(f"{frame}\t{time:.6f}\t" + "\t".join(coords))

    output_path = output_dir / output_name
    with open(output_path, "w") as f:
        for row in header_rows:
            f.write(row + "\n")
        for row in data_rows:
            f.write(row + "\n")

    logger.info(f"  Wrote TRC: {output_path} ({n_frames} frames, {len(marker_names)} markers)")
    return output_path


def parquet_to_fp_mot(
    data_dir: str | Path,
    subject_id: str,
    session: str,
    trial_name: str,
    output_dir: str | Path,
    output_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Extract force plate data from Parquet and write MOT + external loads XML."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = trial_name

    fp_df = _read_forceplates(data_dir, subject_id, session, trial_name)
    if fp_df.is_empty():
        raise ValueError(f"No force plate data for {subject_id}/{session}/{trial_name}")

    fp_names = sorted(fp_df["fp_name"].unique().to_list())
    rate = _detect_rate(fp_df)
    frames = fp_df.select(["time", "frame"]).unique().sort("frame")
    n_frames = len(frames)

    header_rows = [
        f"nRows={n_frames}",
        f"nColumns={1 + len(fp_names) * 9}",
        "endheader",
        "time\t" + "\t".join(
            f"{fp}_{var}_{axis}"
            for fp in fp_names
            for var in ["force", "moment", "cop"]
            for axis in ["x", "y", "z"]
        ),
    ]

    data_rows = []
    for frame_row in frames.iter_rows(named=True):
        frame = frame_row["frame"]
        time = frame_row["time"]
        frame_data = fp_df.filter(pl.col("frame") == frame)

        values = []
        for fp in fp_names:
            for var in ["force", "moment", "cop"]:
                for axis in ["x", "y", "z"]:
                    val = frame_data.filter(
                        (pl.col("fp_name") == fp)
                        & (pl.col("variable") == var)
                        & (pl.col("axis") == axis)
                    )
                    if val.is_empty():
                        values.append("0")
                    else:
                        values.append(f"{float(val['value'].to_list()[0]):.6f}")

        data_rows.append(f"{time:.6f}\t" + "\t".join(values))

    mot_path = output_dir / f"{output_prefix}_forces.mot"
    with open(mot_path, "w") as f:
        for row in header_rows:
            f.write(row + "\n")
        for row in data_rows:
            f.write(row + "\n")

    ext_loads_path = output_dir / f"{output_prefix}_ext_loads.xml"
    _write_ext_loads_xml(ext_loads_path, fp_names, mot_path)

    logger.info(f"  Wrote MOT: {mot_path} ({n_frames} frames, {len(fp_names)} plates)")
    return mot_path, ext_loads_path


def _write_ext_loads_xml(path: Path, fp_names: list[str], mot_path: Path) -> None:
    xml_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<OpenSimDocument Version=\"40000\">",
        "  <ExternalLoads>",
        "    <objects>",
    ]
    for fp_name in fp_names:
        xml_lines.extend([
            f"      <ExternalLoad name=\"{fp_name}\">",
            "        <applied_to_body>calcn_r</applied_to_body>",
            "        <force_expressed_in_body>ground</force_expressed_in_body>",
            "        <point_expressed_in_body>ground</point_expressed_in_body>",
            "        <data_source_name>File</data_source_name>",
            f"        <datafile>{mot_path.name}</datafile>",
            "        <loads_capacity>6</loads_capacity>",
            "      </ExternalLoad>",
        ])
    xml_lines.extend(["    </objects>", "  </ExternalLoads>", "</OpenSimDocument>"])
    with open(path, "w") as f:
        f.write("\n".join(xml_lines))
