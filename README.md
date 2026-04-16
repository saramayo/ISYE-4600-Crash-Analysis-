# ISYE 4600 — AV crash severity (SGO data)

## What runs the models?

- **`scripts/majority_baseline.py`** — majority-class baseline only → `Modeling/majority_baseline_results.csv`
- **`scripts/logistic_regression_baseline.py`** — logistic regression (+ ADS/L2 slices, FN export) → `Modeling/logistic_regression_results.csv`, `lr_coefficients.csv`, `false_negatives.csv`
- **`scripts/02_run_baselines.py`** — runs **both** with **one** load/split and writes the combined `baseline_results.csv` plus the LR artifacts (same as running the two steps yourself).

Shared helpers live in **`scripts/baseline_common.py`** (not executed on its own). Run cleaning first, then either the combined script or the two single-model scripts.

## Project layout

| Path | Purpose |
|------|---------|
| `Data/` | Raw NHTSA SGO CSV exports (four files: ADS/L2 × current/archived). **Input only** — do not edit. |
| `Cleaned/` | **Outputs** from step 1: `sgo_cleaned_incidents.csv`, `data_dictionary.csv`. |
| `Modeling/` | **Outputs** from step 2: see scripts above (`baseline_results.csv` when using `02_run_baselines.py`). |
| `scripts/` | `01_clean_incidents.py`, baselines (`majority_baseline.py`, `logistic_regression_baseline.py`, `02_run_baselines.py`), `baseline_common.py`. |
| `Proposal/` | Written proposal (PDF). |
| `requirements.txt` | Python dependencies. |
| `.venv/` | Local virtual environment (create with `python3 -m venv .venv`). |

## Commands (from this folder)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_clean_incidents.py
.venv/bin/python scripts/02_run_baselines.py
```

Paths inside the scripts use the project root (parent of `scripts/`), so they work from any working directory as long as you invoke the script by path.

## Pipeline order

1. **`01_clean_incidents.py`** — Load raw CSVs → harmonize → one row per incident → labels → writes `Cleaned/`.
2. **Baselines** — `02_run_baselines.py` (both models, one table) **or** `majority_baseline.py` / `logistic_regression_baseline.py` separately. All read `Cleaned/sgo_cleaned_incidents.csv` and write under `Modeling/`.

3. **Presentation figures** — After step 2, run `scripts/make_presentation_figures.py`. It writes PNGs to `Presentation/figures/` and a slide map in `Presentation/SLIDE_FIGURES.txt`.
