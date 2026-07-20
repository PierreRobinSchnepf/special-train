"""Load (or train+save once) the models and expose the dashboard's lookup
methods: predicting one hour of a day J with the correct "which data was
already known at forecast time" rule.

A single `ModelStore` class serves two scopes:
  - **National** (`region_code=None`): dataset_final + 3 models (Kalman, OLS,
    SURE), backtest set, 2025 test. Historical behavior unchanged.
  - **Regional** (`region_code=<int>`): dataset_region_<code> + 2 models
    (Kalman, SURE — OLS is excluded regionally), backtest set, test window
    defined in config.yaml § regional_models.

The dashboard iterates over `store.models` (ordered (key, label) list) rather
than hard-coding the 3 models: adding/removing a model only touches this file.

Assimilation rule (see spec: at J 17:00, forecast J[17-23h] + J+1[0-23h]):
for a target hour `hour`, the last occurrence actually known at "J 17:00" is
today (J) if hour <= 16, else yesterday (J-1).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.dataset import (
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_hourly_equations,
    load_dataset,
    split_train_test,
)
from models.kalman import HourlyKalmanSURModel
from models.ols import HourlyOLSModel
from models.persistence import (
    KALMAN_TRAIN_START,
    OLS_TRAIN_START,
    SURE_TRAIN_START,
    load_artifact,
    regional_artifact_name,
    save_artifact,
)
from models.sure import HourlySUREModel
from src.config import load_config, resolve_path

NATIONAL_TEST_START = "2025-01-01"
NATIONAL_TEST_END = "2026-01-01"

# Model labels (the order defines the display order; the first one is "our
# prediction", shown prominently).
_LABELS = {
    "kalman": "Kalman (our prediction)",
    "ols": "OLS (static)",
    "sure": "SURE (static)",
}

# Predictor grouping by Table 1 block (explainable decomposition).
BLOCKS: dict[str, list[str]] = {
    "Thermal": ["temp_smo", "X1_heating", "X2_smo_heating"],
    "Seasonal (Fourier)": [
        f"{trig}{s}_{grp}" for trig in ("cos", "sin") for s in (1, 2, 3, 4) for grp in ("WD", "WE")
    ],
    "Calendar": ["is_monday", "is_friday", "is_saturday", "is_sunday", "is_end_of_year", "is_off_peak_period"],
}


def _assimilated_date(day_j: dt.date, hour: int) -> dt.date:
    return day_j if hour <= 16 else day_j - dt.timedelta(days=1)


@dataclass
class ModelPred:
    value: float
    lo: float
    hi: float


@dataclass
class HourPrediction:
    date: dt.date
    hour: int
    actual: float | None
    preds: dict[str, ModelPred]          # model key -> (value, CI low, CI high)
    decomposition_log: dict[str, float] = field(default_factory=dict)


class ModelStore:
    def __init__(self, region_code: int | None = None) -> None:
        self.region_code = region_code
        config = load_config()

        if region_code is None:
            # --- national scope ---
            self.label = "National"
            self.model_keys = ["kalman", "ols", "sure"]
            self.test_start, self.test_end = NATIONAL_TEST_START, NATIONAL_TEST_END
            self.df = load_dataset()
            self._artifact = lambda key: f"backtest_{key}"
            self._train_start = {"ols": OLS_TRAIN_START, "sure": SURE_TRAIN_START, "kalman": KALMAN_TRAIN_START}
        else:
            # --- regional scope (OLS excluded) ---
            rm = config["regional_models"]
            names = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
            self.label = f"{region_code} — {names[region_code]}"
            self.model_keys = ["kalman", "sure"]
            self.test_start, self.test_end = rm["test_start"], rm["test_end"]
            ds = resolve_path(config["output"]["processed_dir"]) / rm["dataset_subdir"] / f"dataset_region_{region_code}.parquet"
            self.df = load_dataset(ds)
            self._artifact = lambda key: regional_artifact_name(region_code, key, "backtest")
            ts = rm["train_start"]
            self._train_start = {"sure": ts, "kalman": ts}

        self.models = [(k, _LABELS[k]) for k in self.model_keys]
        self.has_ols = "ols" in self.model_keys

        self.per_hour_all = build_hourly_equations(self.df)
        self.train, self.test = split_train_test(self.per_hour_all, test_start=self.test_start, test_end=self.test_end)
        self.full_per_hour = {h: pd.concat([self.train[h], self.test[h]]).sort_index() for h in range(24)}

        self._load_or_train()

        # Convenience attributes for the CIs (level-space residual variance).
        self._ols_mse = self.ols.mse_resid_ if self.has_ols else None
        self._sure_mse = self.sure.stage1_resid_var_
        self.state_cols = self.kalman.state_cols

    # ------------------------------------------------------------------
    def _load_or_train(self) -> None:
        """Load the artifacts; train+save only the missing ones."""
        self.ols = self.sure = self.kalman = None

        if self.has_ols:
            self.ols = load_artifact(self._artifact("ols"))
            if self.ols is None:
                print(f"[model_store {self.label}] training OLS + saving...")
                train, _ = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["ols"])
                self.ols = HourlyOLSModel().fit(train)
                save_artifact(self.ols, self._artifact("ols"))

        self.sure = load_artifact(self._artifact("sure"))
        if self.sure is None:
            print(f"[model_store {self.label}] training SURE + saving...")
            train, _ = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["sure"])
            self.sure = HourlySUREModel().fit(train)
            save_artifact(self.sure, self._artifact("sure"))

        self.kalman = load_artifact(self._artifact("kalman"))
        if self.kalman is None:
            print(f"[model_store {self.label}] training Kalman + saving...")
            train, test = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["kalman"])
            self.kalman = HourlyKalmanSURModel().fit(train)
            self.kalman.predict(test)  # advance the state over the test set
            save_artifact(self.kalman, self._artifact("kalman"))
        print(f"[model_store {self.label}] ready.")

    # ------------------------------------------------------------------
    def selectable_days(self) -> list[str]:
        """Selectable J days: test-window days present in the panel, leaving a
        one-day margin so that J+1 stays covered."""
        test_dates = sorted(self.test[0].index)
        if not test_dates:
            return []
        return [d.isoformat() for d in test_dates[:-1]]

    def _predictor_row(self, hour: int, date: dt.date) -> pd.Series | None:
        frame = self.full_per_hour[hour]
        if date not in frame.index:
            return None
        row = frame.loc[date]
        if isinstance(row, pd.DataFrame):  # safety net (deduplicated panel)
            row = row.iloc[0]
        return row

    def _apply_what_if(self, row: pd.Series, temp_delta: float) -> pd.Series:
        """Shift the forecast temperature by `temp_delta` °C (approximation:
        shifts the 3 thermal variables directly, without recomputing the EWMA)."""
        if temp_delta == 0.0:
            return row
        row = row.copy()
        row["temp_smo"] = row["temp_smo"] + temp_delta
        row["X1_heating"] = max(0.0, row["X1_heating"] - temp_delta)
        row["X2_smo_heating"] = max(0.0, row["X2_smo_heating"] - temp_delta)
        return row

    # ------------------------------------------------------------------
    def _static_pred(self, beta: np.ndarray, mse_h: float, x_all: np.ndarray) -> ModelPred:
        pred = float(x_all @ beta)
        se = float(np.sqrt(mse_h))
        return ModelPred(pred, pred - 1.96 * se, pred + 1.96 * se)

    def _kalman_pred(self, row: pd.Series, hour: int, day_j: dt.date) -> tuple[ModelPred, dict[str, float]]:
        assim_date = _assimilated_date(day_j, hour)
        traj = self.kalman.full_beta_trajectory(hour)
        lookup_date = min(max(assim_date, traj.index.min()), traj.index.max())
        state = traj.loc[lookup_date].to_numpy(dtype=float)

        x_state = row[self.state_cols].to_numpy(dtype=float)
        contrib = x_state * self.kalman.sur_beta_[hour]     # SUR structural contribution (log)
        pred_log = self.kalman.intercept_[hour] + float(contrib @ state)
        kalman_pred = float(np.expm1(pred_log))

        P = self.kalman.P_state_[hour]
        var_log = float(contrib @ P @ contrib.T + self.kalman.V_[hour])
        se_log = np.sqrt(max(var_log, 0.0))
        pred = ModelPred(
            kalman_pred,
            float(np.expm1(pred_log - 1.96 * se_log)),
            float(np.expm1(pred_log + 1.96 * se_log)),
        )

        decomposition_log = {"Base load (beta0)": self.kalman.intercept_[hour]}
        for block_name, cols in BLOCKS.items():
            idx = [self.state_cols.index(c) for c in cols]
            decomposition_log[block_name] = float(np.sum(contrib[idx] * state[idx]))
        return pred, decomposition_log

    def predict_hour(
        self, day_j: dt.date, target_date: dt.date, hour: int, temp_delta: float = 0.0
    ) -> HourPrediction | None:
        row = self._predictor_row(hour, target_date)
        if row is None:
            return None
        row = self._apply_what_if(row, temp_delta)
        x_all = row[PREDICTOR_COLUMNS].to_numpy(dtype=float)

        preds: dict[str, ModelPred] = {}
        kalman_pred, decomposition_log = self._kalman_pred(row, hour, day_j)
        preds["kalman"] = kalman_pred
        if self.has_ols:
            preds["ols"] = self._static_pred(self.ols.beta_[hour], self._ols_mse[hour], x_all)
        preds["sure"] = self._static_pred(self.sure.beta_[hour], self._sure_mse[hour], x_all)

        actual = None
        ts_candidates = self.full_per_hour[hour]
        if target_date in ts_candidates.index:
            actual_val = ts_candidates.loc[target_date, TARGET_COLUMN]
            actual = float(actual_val) if pd.notna(actual_val) else None

        return HourPrediction(
            date=target_date, hour=hour, actual=actual, preds=preds, decomposition_log=decomposition_log
        )

    def forecast_horizon(self, day_j: dt.date, temp_delta: float = 0.0) -> list[HourPrediction]:
        """31 points: J[17-23h] then J+1[0-23h], as a company would do at J 17:00."""
        results = []
        day_j1 = day_j + dt.timedelta(days=1)
        for hour in range(17, 24):
            pred = self.predict_hour(day_j, day_j, hour, temp_delta)
            if pred is not None:
                results.append(pred)
        for hour in range(24):
            pred = self.predict_hour(day_j, day_j1, hour, temp_delta)
            if pred is not None:
                results.append(pred)
        return results

    def rolling_performance(self, end_date: dt.date, window_days: int = 30) -> pd.DataFrame:
        """For each day d in [end_date-window, end_date], the 24-hour forecast
        of d as it would have been made the day before at 17:00, compared with
        the actuals. Returns a long DataFrame (date, model, rmse, mape) per model."""
        from models.metrics import mape as mape_fn
        from models.metrics import rmse as rmse_fn

        rows = []
        start = end_date - dt.timedelta(days=window_days)
        for d in pd.date_range(start, end_date, freq="D").date:
            j = d - dt.timedelta(days=1)
            preds = [self.predict_hour(j, d, h) for h in range(24)]
            preds = [p for p in preds if p is not None and p.actual is not None]
            if not preds:
                continue
            actual = pd.Series([p.actual for p in preds])
            for key, label in self.models:
                pred_series = pd.Series([p.preds[key].value for p in preds])
                rows.append({
                    "date": d, "model": label,
                    "rmse": rmse_fn(actual, pred_series), "mape": mape_fn(actual, pred_series),
                })
        return pd.DataFrame(rows)
