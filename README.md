# Rat VML Analysis

Analysis of Volumetric Muscle Loss Injury and Treatments in the Rodent Lateral Gastrocnemius.

This repository contains the analysis pipeline and manuscript for a study comparing
biomechanical outcomes across seven treatment groups after VML injury in rats.

## Repository structure

```
scripts/
  run_analysis.py      # Main analysis pipeline (scale → IK → ID → plots)
data/
  raw/                 # Raw motion capture data (C3D/TRC) — not tracked in git
  subjects.csv         # Subject metadata (group, mass, limb lengths)
  ik/                  # IK results (generated)
  id/                  # ID results (generated)
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

# Optionally install MoveDB integration
uv sync --extra movedb

# Import C3D data to .rrd catalog
uv run python scripts/catalog.py import /path/to/sourcedata -o data/rrd/

# Build subjects.csv from catalog (tags subjects with treatment groups)
uv run python scripts/catalog.py subjects data/rrd/ -o data/subjects.csv

# Run the full analysis pipeline
uv run python scripts/run_analysis.py --data-dir data --model ../rat-hindlimb-model/models/osim/rat_hindlimb_bilateral.osim

# Regenerate only figures from cached IK/ID results
uv run python scripts/run_analysis.py --skip-ik --skip-id

# Render the manuscript
quarto render
```

## Dependencies

- **OpenSim 4.6+** (PyPI wheel, Python 3.12–3.13)
- **osimpy** — Pythonic OpenSim tool wrappers (git dependency)
- **tsl-optimization** — tendon slack length optimization (git dependency)
- **spm1d** — Statistical Parametric Mapping for 1D data (group comparisons)

See `pyproject.toml` for the full list.

## Pipeline

1. **Scale** — Subject-specific scaling of the bilateral rat model
2. **IK** — Inverse Kinematics to compute joint angles from marker trajectories
3. **ID** — Inverse Dynamics to compute joint moments from kinematics + GRF
4. **Plot** — Generate kinematics and kinetics comparison figures with SPM
5. **Render** — Build the AGU-formatted manuscript with embedded figures

## CI Pipeline

The GitHub Actions workflow (`pipeline.yml`) runs on push to main and validates
the full analysis pipeline on x86_64 Ubuntu runners.

### Required secrets

| Secret | Purpose | How to set |
|--------|---------|------------|
| `HF_TOKEN` | HuggingFace read token for downloading C3D data | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Add as repo secret |

### What runs on push

1. **validate** — installs deps, validates subject group mapping
2. **prep-data** — downloads C3D data from HuggingFace, builds `subjects.csv`
3. **catalog-import** — imports C3D data to MoveDB `.rrd` catalog with auto-generated group tags
4. **render** — renders the Quarto manuscript

### What runs on manual dispatch

5. **analysis** — full pipeline: scale → IK → ID → figures (requires the model from rat-hindlimb-model)

Go to **Actions → Analysis Pipeline → Run workflow** to trigger the full analysis manually.

## Paper

The manuscript is formatted for submission to AGU journals using the
[Quarto AGU extension](https://github.com/quarto-journals/agu).
