# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are git tags.

## [1.1] — 2026-07-20

Release theme: **cloud deployment + international-ready codebase**.

### Added

- **Cloud deployment stack** ([docs/deployment.md](docs/deployment.md)):
  - `src/storage.py` — S3/MinIO-backed artifact storage (SSP Cloud/Onyxia):
    the deployed app syncs `data/processed/` + `data/models/` (~50 MB) from
    a bucket at boot, once per container; local runs are untouched (no
    secrets → plain disk reads).
  - `scripts/upload_artifacts.py` — one-command push of the runtime
    artifacts to S3 after a local retraining.
  - `.streamlit/secrets.toml.example` — documented secrets template
    (endpoint, keys, bucket, admin password).
  - The dashboard is now **live on Streamlit Community Cloud**.
- **Admin gating** (`dashboard/services/auth.py`): on the public deployment,
  the two write actions — "Check & refresh" (heavy download + retraining)
  and "Run a live forecast" (external API calls + DB writes) — are hidden
  behind a sidebar admin password. Locally everything stays unlocked.
- **Regional artifacts for all 12 regions**: added the missing Centre-Val
  de Loire weather station (dept 45, Orléans) to the basket and generated
  the full regional set (12 datasets, 48 model artifacts) powering the map
  of France in the deployed app.
- **Docs**: `docs/data-dictionary.md`, `docs/design-decisions.md`,
  `docs/deployment.md`.
- **Release hygiene**: MIT `LICENSE` file (the README previously pointed to
  a nonexistent file), GitHub Actions CI running the test suite on every
  push, this changelog.
- **README**: new "The mathematics" section — feature-engineering formulas,
  OLS/SURE/FGLS derivation sketch, the Kalman-SUR state-space model — with
  figures exported from the (re-executed) notebooks.
- Dev container configuration (`.devcontainer/`) for Codespaces.

### Changed

- **Entire project translated to English**: docstrings, comments, CLI help,
  dashboard UI, notebooks (markdown + code), `config.yaml` comments, docs.
  French identifiers renamed (`modele`→`model`, `Réel`→`Actual`,
  `heure`→`hour`, `reel`→`actual`, `a_verifier`→`pending`,
  `heure_locale`→`local_hour`, `source_meteo`→`weather_source`,
  `erreur_abs_*`→`abs_error_*`).
- **Repository layout cleaned up**: the five root pipeline scripts moved to
  `scripts/` (`fetch_data`, `build_dataset`, `build_regional_dataset`,
  `train_models`, `train_regional_models`); French docs replaced by the
  `docs/` folder; `README_AI.md`, `data_dictionary.md` (superseded) and the
  placeholder demo GIF removed.
- Notebooks re-executed end-to-end after translation (all figures now carry
  English labels).
- README hero image is now a real model output (Kalman vs SUR vs actual,
  January 2025) instead of a placeholder GIF.

### Fixed

- `tracking.sqlite3` schema migration: the live-forecast store transparently
  renames the v1.0 French columns (`heure_locale`, `source_meteo`) and
  values (`prevue`/`observee` → `forecast`/`observed`) on first open — old
  recorded forecasts remain readable.
- `scripts/upload_artifacts.py` no longer crashes on Windows consoles
  (cp1252) — arrow characters replaced with ASCII log prefixes.
- Removed dead code (unused `forecast_cutoff` in `pipeline/real_forecast.py`).

## [1.0] — 2026-07-18

Initial release: hourly dataset pipeline (2018–present), OLS/SURE/Kalman-SUR
benchmark, regional forecasting (12 gas regions), live one-day-ahead
pipeline with quality tracking, Streamlit dashboard (map of France,
forecasts, benchmark, monitoring).

[1.1]: https://github.com/PierreRobinSchnepf/special-train/compare/v1.0...v1.1
[1.0]: https://github.com/PierreRobinSchnepf/special-train/releases/tag/v1.0
