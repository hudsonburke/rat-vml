# Rat VML Analysis

Analysis of Volumetric Muscle Loss Injury and Treatments in the Rodent Lateral Gastrocnemius.

This repository contains the analysis pipeline and manuscript for a study comparing
biomechanical outcomes across seven treatment groups after VML injury in rats.

## Repository structure

```
scripts/
  ingest.py            # One-time C3D → .rrd conversion (workstation only)
  catalog.py           # MoveDB catalog queries (subjects, trials, events)
  run_analysis.py      # Main analysis pipeline (scale → IK → ID → plots)
src/rat_vml/analysis/  # Analysis module (events, forces, io, pipeline, plots, queries)
data/
  rrd/                 # Pre-built .rrd catalog (downloaded from HuggingFace)
  subjects.csv         # Subject metadata (group, mass, limb lengths) — generated
  results/             # IK/ID results per subject (generated)
  figures/             # Output figures (generated)
images/                # Manuscript figures (committed for Quarto render)
_extensions/           # AGU journal Quarto extension
```

## Quickstart

```shell
git clone --recurse-submodules https://github.com/hudsonburke/rat-vml.git
cd rat-vml

# Install dependencies
uv sync

# Download pre-built .rrd catalog from HuggingFace
uv run python scripts/ingest.py pull

# Build subjects.csv from catalog (tags subjects with treatment groups)
uv run python scripts/catalog.py subjects data/rrd/ -o data/subjects.csv

# Run the full analysis pipeline
uv run python scripts/run_analysis.py --data-dir data --model ../rat-hindlimb-model/models/osim/rat_hindlimb_bilateral.osim

# Render the manuscript
quarto render
```

## C3D Ingestion (one-time, workstation only)

The `.rrd` catalog files are pre-built and stored in the HuggingFace dataset.
To rebuild them from raw C3D data (requires x86_64 machine with ezc3d):

```shell
# Install workstation-only deps
uv sync --extra ingest

# Convert C3D to .rrd
uv run python scripts/ingest.py convert --c3d-dir /path/to/sourcedata -o data/rrd/

# Push updated .rrd files to HuggingFace
uv run python scripts/ingest.py push --rrd-dir data/rrd/
```

## Dependencies

- **OpenSim 4.6+** (PyPI wheel, Python 3.12–3.13)
- **osimpy** — Pythonic OpenSim tool wrappers (git dependency)
- **tsl-optimization** — tendon slack length optimization (git dependency)
- **rathindlimb** — rat model scaling code (git dependency)
- **spm1d** — Statistical Parametric Mapping for 1D data (group comparisons)

See `pyproject.toml` for the full list.

## Pipeline

```
.rdd catalog ← DuckDB query → find valid walking trials (7 events, no gaps)
       ↓
For selected trials: extract markers/forces from .rrd → write TRC/MOT
       ↓
Scale → IK → ID → spline to stance+swing → group aggregation → figures
```

The `.rrd` files are the single source of truth. No C3D reading after ingestion.

## CI Pipeline

The GitHub Actions workflow (`pipeline.yml`) runs on push to main.

### Required secrets

| Secret | Purpose | How to set |
|--------|---------|------------|
| `HF_TOKEN` | HuggingFace read token for downloading .rrd files | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Add as repo secret |

### What runs on push

1. **validate** — installs deps, validates subject group mapping and module imports
2. **catalog** — downloads `.rrd` files from HuggingFace, builds `subjects.csv`, queries for valid trials
3. **render** — renders the Quarto manuscript (non-blocking)

### What runs on manual dispatch

4. **analysis** — full pipeline: scale → IK → ID → figures (requires the model from rat-hindlimb-model)

Go to **Actions → Analysis Pipeline → Run workflow** to trigger the full analysis manually.

## Paper

The manuscript is formatted for submission to AGU journals using the
[Quarto AGU extension](https://github.com/quarto-journals/agu).
