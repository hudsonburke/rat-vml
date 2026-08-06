"""Tests for Parquet -> OpenSim I/O module."""

import polars as pl
import pytest
from pathlib import Path
from rat_vml.analysis.parquet_io import parquet_to_trc, parquet_to_fp_mot


@pytest.fixture
def sample_markers(tmp_path):
    """Create sample marker Parquet data."""
    markers = pl.DataFrame({
        "time": [0.0, 0.005, 0.01, 0.015, 0.02],
        "frame": [0, 1, 2, 3, 4],
        "marker_name": ["r_asis"] * 5,
        "x": [10.0, 10.1, 10.2, 10.3, 10.4],
        "y": [20.0, 20.1, 20.2, 20.3, 20.4],
        "z": [30.0, 30.1, 30.2, 30.3, 30.4],
        "trial_name": ["Walk01"] * 5,
        "subject_id": ["A01"] * 5,
        "session_id": ["Baseline"] * 5,
    })
    (tmp_path / "A01").mkdir()
    markers.write_parquet(tmp_path / "A01" / "markers.parquet")
    return tmp_path


@pytest.fixture
def sample_forceplates(tmp_path):
    """Create sample force plate Parquet data."""
    fps = pl.DataFrame({
        "time": [0.0, 0.0005, 0.001],
        "frame": [0, 1, 2],
        "fp_name": ["FP1"] * 3,
        "variable": ["force"] * 3,
        "axis": ["x", "y", "z"],
        "value": [100.0, 200.0, 300.0],
        "trial_name": ["Walk01"] * 3,
        "subject_id": ["A01"] * 3,
        "session_id": ["Baseline"] * 3,
    })
    (tmp_path / "A01").mkdir(exist_ok=True)
    fps.write_parquet(tmp_path / "A01" / "forceplates.parquet")
    return tmp_path


def test_parquet_to_trc(sample_markers, tmp_path):
    trc_path = parquet_to_trc(sample_markers, "A01", "Baseline", "Walk01", tmp_path)
    assert trc_path.exists()
    assert trc_path.name == "Walk01.trc"

    with open(trc_path) as f:
        lines = f.readlines()

    # Header lines
    assert "Walk01.trc" in lines[0]
    assert "NumFrames" in lines[2]
    assert "r_asis" in lines[3]
    assert "X\tY\tZ" in lines[4]

    # Data lines
    assert len(lines) == 5 + 5  # 5 header + 5 data rows
    assert "1\t" in lines[5]  # Frame 1


def test_parquet_to_trc_custom_name(sample_markers, tmp_path):
    trc_path = parquet_to_trc(sample_markers, "A01", "Baseline", "Walk01", tmp_path, output_name="custom.trc")
    assert trc_path.name == "custom.trc"


def test_parquet_to_trc_missing_data(sample_markers, tmp_path):
    with pytest.raises(ValueError, match="No marker data"):
        parquet_to_trc(sample_markers, "A01", "Baseline", "Walk99", tmp_path)


def test_parquet_to_fp_mot(sample_forceplates, tmp_path):
    mot_path, ext_loads_path = parquet_to_fp_mot(sample_forceplates, "A01", "Baseline", "Walk01", tmp_path)
    assert mot_path.exists()
    assert ext_loads_path.exists()
    assert mot_path.name == "Walk01_forces.mot"
    assert ext_loads_path.name == "Walk01_ext_loads.xml"

    with open(mot_path) as f:
        lines = f.readlines()

    # Header
    assert "nRows=" in lines[0]
    assert "nColumns=" in lines[1]
    assert "endheader" in lines[2]
    assert "FP1" in lines[3]

    # Data lines
    assert len(lines) == 4 + 3  # 4 header + 3 data rows


def test_parquet_to_fp_mot_custom_prefix(sample_forceplates, tmp_path):
    mot_path, ext_loads_path = parquet_to_fp_mot(sample_forceplates, "A01", "Baseline", "Walk01", tmp_path, output_prefix="custom")
    assert mot_path.name == "custom_forces.mot"
    assert ext_loads_path.name == "custom_ext_loads.xml"


def test_parquet_to_fp_mot_missing_data(sample_forceplates, tmp_path):
    with pytest.raises(ValueError, match="No force plate data"):
        parquet_to_fp_mot(sample_forceplates, "A01", "Baseline", "Walk99", tmp_path)


def test_ext_loads_xml_content(sample_forceplates, tmp_path):
    _, ext_loads_path = parquet_to_fp_mot(sample_forceplates, "A01", "Baseline", "Walk01", tmp_path)
    with open(ext_loads_path) as f:
        content = f.read()

    assert "<?xml" in content
    assert "ExternalLoads" in content
    assert "FP1" in content
    assert "calcn_r" in content
    assert "Walk01_forces.mot" in content
