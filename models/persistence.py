"""Saving/loading of trained models, to avoid retraining everything (~40s:
OLS + SURE + Kalman) on every dashboard launch.

Two model sets, with different training windows:
  - "backtest"   : used by the Forecast/Benchmark/Monitoring tabs. OLS/SURE
    trained on [2020-01-01, 2025-01-01), Kalman on [2018-01-01, 2025-01-01)
    with its state then advanced over the 2025 test — the 2025 test stays
    held out.
  - "production" : used by the Live pipeline tab. Same lower bound, but
    trained through the end of 2025 (no held-out test) — the requested
    "latest retraining including 2025 data".

`scripts/train_models.py` builds these artifacts; `dashboard/services/
model_store.py` and `pipeline/real_forecast.py` load them, and only fall
back to on-the-fly retraining when the cache is absent.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "data" / "models"

OLS_TRAIN_START = "2020-01-01"     # OLS/SURE: 2018-2019 excluded (too old)
SURE_TRAIN_START = "2020-01-01"
KALMAN_TRAIN_START = "2018-01-01"  # Kalman: full history, unchanged

BACKTEST_TEST_START = "2025-01-01"
BACKTEST_TEST_END = "2026-01-01"

# --- Regional models (Step C/D) --------------------------------------------
# One set per region and per usage. File name:
#   data/models/regional/region<code>_<model>_<set>.pkl
# set ∈ {backtest (held-out test, for metrics/replay dashboard),
#        production (trained through the latest data, for forecasting)}.
REGIONAL_SUBDIR = "regional"


def regional_artifact_name(region_code: int, model: str, artifact_set: str = "backtest") -> str:
    return f"{REGIONAL_SUBDIR}/region{region_code}_{model}_{artifact_set}"


def _strip_statsmodels_results(obj: Any) -> None:
    """Strip the training data (exog/endog) from the statsmodels
    RegressionResults objects buried inside our models before saving. Without
    this, each .pkl embeds the full design matrix — up to ~700 MB for the
    SURE system (whitened_result_ carries a (24*T, 24*k) matrix). Verified:
    `.predict(X)` with an explicit X (our only post-training usage) returns
    identical results before/after `remove_data()`; only `.predict()` without
    arguments breaks (never used here, an explicit `X` is always passed)."""
    # Deferred imports: avoids a cycle (models/__init__ imports persistence
    # indirectly through dataset.py, which is imported by ols/sure/kalman).
    from models.kalman import HourlyKalmanSURModel
    from models.ols import HourlyOLSModel
    from models.sure import HourlySUREModel

    if isinstance(obj, HourlyOLSModel):
        for res in obj.results_.values():
            res.remove_data()
    elif isinstance(obj, HourlySUREModel):
        for res in obj.stage1_results_.values():
            res.remove_data()
        if obj.whitened_result_ is not None:
            obj.whitened_result_.remove_data()
    elif isinstance(obj, HourlyKalmanSURModel):
        if obj.sure_ is not None:
            _strip_statsmodels_results(obj.sure_)


def save_artifact(obj: Any, name: str) -> Path:
    _strip_statsmodels_results(obj)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_artifact(name: str) -> Any | None:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
