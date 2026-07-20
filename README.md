<div align="center">

# gas-hourly-forecast

**Hourly forecasting of French natural gas consumption, powered by a dynamically-adjusted Kalman/SUR model.**

[![Live demo](https://img.shields.io/badge/demo-streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://gas-hourly-forecast.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![Tests](https://img.shields.io/badge/tests-passing-2ea44f)](#tests)
[![Status](https://img.shields.io/badge/status-active-4fb5c7)](#)

[**🔗 Try the live dashboard →**](https://gas-hourly-forecast.streamlit.app/)

<img src="assets/demo-placeholder.gif" width="720" alt="Dashboard demo placeholder — replace with a real screen recording">

</div>

---

## Overview

`gas-hourly-forecast` builds an hourly, gap-free dataset of French natural gas consumption (2018–present) and benchmarks three forecasting models — hourly OLS, SURE (Zellner), and a Kalman-adjusted SUR — to predict `y_gas_mw` one day ahead. An interactive Streamlit dashboard exposes forecasts, backtests, and a live "real pipeline" fed by fresh weather and consumption data, at both national and regional (12 gas regions) granularity.

**[→ Live demo on Streamlit](https://gas-hourly-forecast.streamlit.app/)**

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data sources](#data-sources)
- [Modeling](#modeling)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Roadmap](#roadmap)
- [License](#license)

## Features

- **Gap-free hourly dataset (2018–present)** — UTC-indexed pipeline merging gas consumption, hourly temperature, public holidays, and school holidays.
- **Three benchmarked models** — independent hourly OLS, joint SURE (FGLS), and a Kalman filter that lets SUR coefficients drift over time.
- **National + 12 regional forecasts** — the same model stack, parameterized per gas region.
- **Live pipeline** — a daily job pulls real weather and consumption data and produces genuine one-day-ahead forecasts, tracked against actuals once they land.
- **Interactive dashboard** — forecast curves with confidence intervals, a weather what-if tool, day-by-day benchmark view, and rolling accuracy tracking.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# Build the full dataset (2018 -> today)
python build_dataset.py

# Quick 1-month sample (reuses cache if already downloaded)
python build_dataset.py --sample-months 1

# Train and persist all model artifacts
python train_models.py

# Regional pipeline (12 gas regions)
python fetch_data.py --only gas_regional
python build_regional_dataset.py
python train_regional_models.py

# Run the dashboard locally
streamlit run dashboard/app.py

# Run today's real forecast
python -m pipeline.run_daily
```

Output: `data/processed/dataset_final.parquet` + a QC report (NaN rates, duplicates, index gaps).

## Data sources

| Source | Provider | Native frequency | Timezone |
|---|---|---|---|
| Gas consumption | ODRÉ (`consommation-quotidienne-brute`) | hourly | UTC |
| Temperature | Météo-France (hourly climatological data, by department) | hourly | UTC (SYNOP) |
| Public holidays | data.gouv.fr / etalab | daily | Europe/Paris |
| School holidays | data.education.gouv.fr (zones A/B/C) | intervals | Europe/Paris |
| Near-real-time weather | Open-Meteo (AROME model, no key required) | hourly | UTC |

The dataset is indexed in continuous UTC to sidestep Europe/Paris DST transitions entirely; calendar-dependent features are derived by converting timestamps to local time only to extract the civil date, never by re-indexing on it.

## Modeling

Three models forecast each of the 24 local hours independently, from the same feature set (thermal inertia, Fourier seasonality, calendar effects):

- **Hourly OLS** — 24 fully independent regressions.
- **SURE** (Zellner, 1962) — the same system estimated jointly via FGLS, exploiting same-day correlation across hourly residuals.
- **Kalman-adjusted SUR** — each SUR coefficient gets a multiplicative scale factor estimated by a Kalman filter, letting effects drift over time while keeping the SUR structure explainable.

Trained on 2018–2024, tested on full-year 2025. Across the 12 gas regions, the Kalman model beats SURE on RMSE by 16% on average (regional MAPE: Kalman 7.6% vs. SURE 12.0%).

Pre-trained artifacts live in `data/models/`; the dashboard and pipeline load them directly rather than retraining on every run.

## Project structure

```
config.yaml               # single source of truth for parameters/thresholds/URLs
fetch_data.py              # step 1: raw ingestion, idempotent, -> data/raw/
build_dataset.py           # steps 2-5: alignment, features, QC, export
src/                       # feature engineering (thermal, Fourier, calendar)
models/                    # OLS, SURE, Kalman implementations + persistence
dashboard/                 # Streamlit app
pipeline/                  # daily real-world forecasting job
notebooks/                 # exploration & benchmark notebooks
tests/                     # unit tests
data_dictionary.md         # full column-by-column reference
```

## Tests

```bash
pytest tests/ -v
```

Covers feature formulas (EWMA smoothing, thermal clipping, Fourier weekday/weekend masking), model correctness on synthetic data (OLS/SURE coefficient recovery, FGLS whitening), and the Kalman filter's behavior on synthetic sequences.

## Roadmap

- [ ] Automated daily scheduling for the real pipeline (currently manual)
- [ ] Mid-August consumption dip as an explicit feature
- [ ] Extend regional coverage below the department level

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built with Streamlit, statsmodels, and a Kalman filter. Not affiliated with GRTgaz, Teréga, or Météo-France.</sub>
</div>
