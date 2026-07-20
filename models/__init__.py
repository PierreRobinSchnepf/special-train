"""Benchmark models for hourly gas consumption forecasting.

Two approaches, both structured as 24 equations (one per Europe/Paris local
hour), sharing the same predictors (Table 1 of the source report):

- `models.ols.HourlyOLSModel`   : 24 independent OLS regressions.
- `models.sure.HourlySUREModel` : the same system estimated jointly by FGLS
  (Seemingly Unrelated Regressions, Zellner 1962), exploiting the
  contemporaneous correlation of residuals across equations (same day).

See `models.dataset` for the balanced day x hour panel preparation and the
train/test split, and `models.metrics` for RMSE/MAPE.
"""
from models.dataset import (
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_hourly_equations,
    load_dataset,
    split_train_test,
)
from models.kalman import HourlyKalmanSURModel
from models.ols import HourlyOLSModel
from models.sure import HourlySUREModel

__all__ = [
    "PREDICTOR_COLUMNS",
    "TARGET_COLUMN",
    "load_dataset",
    "build_hourly_equations",
    "split_train_test",
    "HourlyOLSModel",
    "HourlySUREModel",
    "HourlyKalmanSURModel",
]
