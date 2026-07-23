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

# Place motion data in data/raw/ and create data/subjects.csv, then:
uv run python scripts/run_analysis.py

# Regenerate only figures from cached IK/ID results:
uv run python scripts/run_analysis.py --skip-ik --skip-id

# Process a single treatment group:
uv run python scripts/run_analysis.py --group NR

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

## Paper

The manuscript is formatted for submission to AGU journals using the
[Quarto AGU extension](https://github.com/quarto-journals/agu).
