"""Tests for ParquetCatalog query module."""

import polars as pl
import pytest
from pathlib import Path
from rat_vml.analysis.parquet_catalog import ParquetCatalog


@pytest.fixture
def sample_data(tmp_path):
    markers_a01 = pl.DataFrame({
        "time": [0.0, 0.005, 0.01, 0.015, 0.02] * 4,
        "frame": list(range(20)),
        "marker_name": ["r_asis"] * 5 + ["l_asis"] * 5 + ["r_knee"] * 5 + ["l_knee"] * 5,
        "x": [1.0] * 20, "y": [2.0] * 20, "z": [3.0] * 20,
        "trial_name": ["Walk01"] * 10 + ["Walk02"] * 10,
        "subject_id": ["A01"] * 20, "session_id": ["Baseline"] * 20,
    })
    events_a01 = pl.DataFrame({
        "context": ["Left", "Right", "Left", "Right", "Left", "Right", "Left"] * 2,
        "label": ["Foot Strike", "Foot Strike", "Foot Off", "Foot Off", "Foot Strike", "Foot Strike", "Foot Off"] * 2,
        "time": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] * 2,
        "trial_name": ["Walk01"] * 7 + ["Walk02"] * 7,
        "subject_id": ["A01"] * 14, "session_id": ["Baseline"] * 14,
        "group": ["Control"] * 14,
    })
    markers_b02 = pl.DataFrame({
        "time": [0.0, 0.005, 0.01], "frame": [0, 1, 2],
        "marker_name": ["r_asis"] * 3, "x": [1.0] * 3, "y": [2.0] * 3, "z": [3.0] * 3,
        "trial_name": ["Walk01"] * 3, "subject_id": ["B02"] * 3, "session_id": ["Baseline"] * 3,
    })
    events_b02 = pl.DataFrame({
        "context": ["Left", "Right"], "label": ["Foot Strike", "Foot Strike"],
        "time": [0.1, 0.15], "trial_name": ["Walk01"] * 2,
        "subject_id": ["B02"] * 2, "session_id": ["Baseline"] * 2, "group": ["TEMR"] * 2,
    })
    (tmp_path / "A01").mkdir()
    markers_a01.write_parquet(tmp_path / "A01" / "markers.parquet")
    events_a01.write_parquet(tmp_path / "A01" / "events.parquet")
    (tmp_path / "B02").mkdir()
    markers_b02.write_parquet(tmp_path / "B02" / "markers.parquet")
    events_b02.write_parquet(tmp_path / "B02" / "events.parquet")
    return tmp_path


def test_all_subjects(sample_data):
    cat = ParquetCatalog(sample_data)
    subjects = cat.all_subjects()
    assert len(subjects) == 2
    assert set(subjects["subject_id"].to_list()) == {"A01", "B02"}


def test_subjects_by_group(sample_data):
    cat = ParquetCatalog(sample_data)
    control = cat.subjects_by_group("Control")
    assert len(control) == 1
    assert control["subject_id"][0] == "A01"
    temr = cat.subjects_by_group("TEMR")
    assert len(temr) == 1
    assert temr["subject_id"][0] == "B02"


def test_all_trials(sample_data):
    cat = ParquetCatalog(sample_data)
    trials = cat.all_trials()
    assert len(trials) == 3


def test_trials_by_session(sample_data):
    cat = ParquetCatalog(sample_data)
    baseline = cat.trials_by_session("Baseline")
    assert len(baseline) == 3
    week24 = cat.trials_by_session("Week24")
    assert len(week24) == 0


def test_valid_walking_trials_min7(sample_data):
    cat = ParquetCatalog(sample_data)
    valid = cat.valid_walking_trials(min_events=7)
    assert len(valid) == 2
    assert all(s == "A01" for s in valid["subject_id"].to_list())


def test_valid_walking_trials_min2(sample_data):
    cat = ParquetCatalog(sample_data)
    valid = cat.valid_walking_trials(min_events=2)
    assert len(valid) == 3


def test_empty_catalog(tmp_path):
    cat = ParquetCatalog(tmp_path)
    assert len(cat.all_subjects()) == 0
    assert len(cat.all_trials()) == 0
