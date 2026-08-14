# Rat VML Analysis

Analysis of Volumetric Muscle Loss Injury and Treatments in the Rodent Lateral Gastrocnemius.

This repository contains the analysis pipeline and manuscript for a study comparing
biomechanical outcomes across seven treatment groups after VML injury in rats.

## Repository Structure

```
scripts/
  scale.py             # Scale model to subject anthropometrics
  run_ik.py            # Run Inverse Kinematics
  run_id.py            # Run Inverse Dynamics
  run_moco.py          # Run MocoInverse muscle analysis
  aggregate.py         # Group aggregation and comparison
  compare.py           # Statistical comparisons between groups
  plot_helpers.py      # Plotting utilities
src/rat_vml/analysis/
  scaling.py           # Model scaling wrapper
  filtering.py         # Marker and force plate filtering (Butterworth, notch)
  events.py            # Gait event validation
  static_trial.py      # Static trial selection for scaling
  parquet_io.py        # Parquet → TRC/MOT export
  results_storage.py   # Write IK/ID/Moco outputs to Parquet
  aggregation.py       # Group aggregation and SPM t-tests
  plots.py             # Manuscript-quality figures
  subject_groups.py    # Subject-to-treatment-group mapping
notebooks/
  explore.py           # Marimo notebook for data exploration
```

## Quickstart

```bash
git clone https://github.com/hudsonburke/rat-vml.git
cd rat-vml

# Install dependencies
uv sync

# Download data from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download('hudsonburke/rat-hindlimb-mocap', repo_type='dataset',
                  local_dir='data', allow_patterns=['processed/**/*.parquet'])
"

# Scale model for a subject
python scripts/scale.py \
  --data-dir data/processed \
  --subject BAA01 \
  --session baseline \
  --model ~/rat-hindlimb-model/models/osim/rat_hindlimb_bilateral.osim

# Run IK on a valid walking trial
python scripts/run_ik.py \
  --data-dir data/processed \
  --subject BAA01 \
  --session baseline \
  --trial Walk02 \
  --model data/results/BAA01/BAA01_baseline_scaled.osim

# Run ID
python scripts/run_id.py \
  --data-dir data/processed \
  --subject BAA01 \
  --session baseline \
  --trial Walk02 \
  --model data/results/BAA01/BAA01_baseline_scaled.osim \
  --ik-file data/results/BAA01/ik_baseline_Walk02.mot

# Explore data interactively
marimo edit notebooks/explore.py
```

## Data Pipeline

```
C3D files (rat-hindlimb-mocap)
  → movedb ingestion → Parquet files
  → rat-vml reads via movedb.catalog.MoveDB
  → Filter → TRC/MOT → Scale → IK → ID → Results
```

### Data Access

```python
from movedb import MoveDB

db = MoveDB(Path("data/processed"))

# Load markers, force plates, events
markers = db.get_points("BAA01", session="baseline")
forceplates = db.get_forceplates("BAA01", session="baseline")
events = db.get_events("BAA01", session="baseline")

# Get session parameters (mass, bone lengths)
params = db.get_parameters("BAA01", session="baseline")

# SQL queries across all subjects
df = db.query("SELECT subject_id, session_id, mass FROM parameters")

# List available data
subjects = db.subjects()
sessions = db.sessions("BAA01")
trials = db.trials("BAA01", session="baseline")
```

## Dependencies

- **movedb-core** — Data ingestion and catalog (git dependency)
- **osimpy** — OpenSim tool wrappers (git dependency)
- **rathindlimb** — Rat model scaling (git dependency)
- **spm1d** — Statistical Parametric Mapping
- **marimo** — Interactive notebooks (optional)

See `pyproject.toml` for the full list.

## Interactive Exploration

Launch the marimo notebook for interactive data exploration:

```bash
marimo edit notebooks/explore.py
```

This provides:
- Browse subjects, sessions, and trials
- Plot marker trajectories and force plate data
- Inspect gait events
- Validate walking trial selection
- View session parameters (mass, bone lengths)

## Manuscript

The manuscript is formatted for submission to AGU journals using the
[Quarto AGU extension](https://github.com/quarto-journals/agu).

```bash
quarto render
```
