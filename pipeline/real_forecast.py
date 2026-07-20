"""Live-pipeline orchestrator: at the present moment, produce a forecast for
J (17:00) -> J+1 (23:00) using only the data genuinely available now.

Three departures from the backtest in `dashboard/services/model_store.py`
(which replays a history where everything is already known):
  1. The official gas target lags by ~45-50 days: the gap is filled up to
     "day G" (~15-20 days back) with `pipeline.gas_freshness` (regional
     industrial+distribution reconstruction).
  2. Beyond day G there is no ground truth: the Kalman state is frozen at
     its day-G value and propagated (random walk, no update) — the same
     principle as the backtest's trajectory lookup, not a new statistical
     idea.
  3. Weather between day G and today is real (already observed); only the
     weather from today to J+1 is a true forecast
     (`pipeline.weather_forecast`, Open-Meteo).

The result includes explicit freshness metadata (day G, which hours rely on
observed vs forecast weather) so the dashboard never presents a forecast as
more "real" than it is.
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
from models.persistence import KALMAN_TRAIN_START, OLS_TRAIN_START, SURE_TRAIN_START, load_artifact, save_artifact
from models.sure import HourlySUREModel
from pipeline.gas_freshness import fetch_fresh_gas_total, last_available_day
from pipeline.weather_forecast import continue_temp_smo, fetch_national_temperature
from src.calendar_features import compute_end_of_year_flag, compute_off_peak_flag, compute_weekday_flags
from src.fourier_features import compute_fourier_features
from src.thermal_features import compute_x1_heating, compute_x2_smo_heating

CALENDAR_TZ = "Europe/Paris"
STATE_COLS = [c for c in PREDICTOR_COLUMNS if c != "beta_0"]


@dataclass
class RealForecastResult:
    day_j: dt.date
    day_g: dt.date               # last day of assimilated gas ground truth
    generated_at: dt.datetime
    horizon: pd.DataFrame          # timestamp, local_hour, kalman/ols/sure, weather_source (observed/forecast)
    warnings: list[str] = field(default_factory=list)


def _build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Recompute the Fourier/calendar blocks for an arbitrary UTC index
    (recent past or future), reusing the public/school holidays already
    cached by `fetch_data.py` (see the documented limitation in the warnings
    if the horizon exceeds their coverage)."""
    from scripts.build_dataset import load_holidays, load_school_holidays
    from src.config import load_config as load_pipeline_config

    config = load_pipeline_config()
    fourier_cfg = config["fourier"]

    fourier_df = compute_fourier_features(index, fourier_cfg["harmonics"], fourier_cfg["days_in_year"], CALENDAR_TZ)
    weekday_flags = compute_weekday_flags(index, CALENDAR_TZ)
    end_of_year = compute_end_of_year_flag(index, CALENDAR_TZ, config["calendar"]["end_of_year_window"])

    holiday_dates = load_holidays(config)
    school_dates = load_school_holidays(config)
    off_peak = compute_off_peak_flag(index, CALENDAR_TZ, holiday_dates, school_dates, end_of_year)

    beta_0 = pd.Series(1, index=index, name="beta_0")
    return pd.concat([fourier_df, weekday_flags, end_of_year, off_peak, beta_0], axis=1)


def _to_per_hour(df: pd.DataFrame, target_col: str | None = None) -> dict[int, pd.DataFrame]:
    """Like `models.dataset.build_hourly_equations`, but without requiring a
    non-null target (useful for the future horizon, where none exists) and
    without balancing across hours (unnecessary, we only predict)."""
    local = df.index.tz_convert(CALENDAR_TZ)
    work = df.copy()
    if target_col is None or target_col not in work.columns:
        work[TARGET_COLUMN] = np.nan
        target_col = TARGET_COLUMN
    work["local_date"] = local.date
    work["local_hour"] = local.hour
    work["utc_ts"] = df.index
    return {h: work[work["local_hour"] == h].set_index("local_date").sort_index() for h in range(24)}


def _direct_kalman_predict(model: HourlyKalmanSURModel, per_hour: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
    """Prediction with a FROZEN state (never assimilated/updated) — the only
    valid operation on a horizon without ground truth:
    `HourlyKalmanSURModel.predict()` would run the internal Kalman update loop
    even with update_state=False on the caller side, corrupting the state
    with NaN targets. So H_t . frozen_state is computed directly, exactly as
    the dashboard backtest does."""
    preds = {}
    for h in range(24):
        frame = per_hour[h]
        if frame.empty:
            preds[h] = pd.Series(dtype=float)
            continue
        x = frame[STATE_COLS].to_numpy(dtype=float)
        contrib = x * model.sur_beta_[h][None, :]
        pred_log = model.intercept_[h] + contrib @ model.beta_state_[h]
        preds[h] = pd.Series(np.expm1(pred_log), index=frame["utc_ts"].to_numpy())
    return preds


def _static_predict(beta_by_hour, predictor_cols: list[str], per_hour: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
    preds = {}
    for h in range(24):
        frame = per_hour[h]
        if frame.empty:
            preds[h] = pd.Series(dtype=float)
            continue
        x = frame[predictor_cols].to_numpy(dtype=float)
        preds[h] = pd.Series(x @ beta_by_hour[h], index=frame["utc_ts"].to_numpy())
    return preds


def run_real_forecast(now: dt.datetime | None = None) -> RealForecastResult:
    warnings: list[str] = []
    now = now or dt.datetime.now(dt.timezone.utc)
    day_j = now.astimezone().date()  # local civil day "today"
    day_j1 = day_j + dt.timedelta(days=1)

    # --- 1. Pre-trained "production" models (train_models.py --only production):
    #        OLS/SURE >= 2020, Kalman >= 2018, all trained through end of 2025
    #        (no held-out test — "one last retraining including the 2025 data"). ---
    df = load_dataset()
    per_hour_all = build_hourly_equations(df)

    ols = load_artifact("production_ols")
    sure = load_artifact("production_sure")
    kalman = load_artifact("production_kalman")

    if ols is None or sure is None or kalman is None:
        warnings.append(
            "'production' artifacts missing from data/models/ — training on the fly "
            "(~40s). Run `python scripts/train_models.py` to avoid this cost on every run."
        )
        if ols is None:
            ols_train, _ = split_train_test(per_hour_all, "2026-01-01", "2026-01-01", train_start=OLS_TRAIN_START)
            ols = HourlyOLSModel().fit(ols_train)
            save_artifact(ols, "production_ols")
        if sure is None:
            sure_train, _ = split_train_test(per_hour_all, "2026-01-01", "2026-01-01", train_start=SURE_TRAIN_START)
            sure = HourlySUREModel().fit(sure_train)
            save_artifact(sure, "production_sure")
        if kalman is None:
            kalman_train, _ = split_train_test(per_hour_all, "2026-01-01", "2026-01-01", train_start=KALMAN_TRAIN_START)
            kalman = HourlyKalmanSURModel().fit(kalman_train)
            save_artifact(kalman, "production_kalman")

    # --- 2. Catch up on the "stub" beyond the production training window
    #        (e.g. 2026-01-01 -> last day of dataset_final.parquet) ---
    last_dataset_date = df.index.max().date()
    stub = {
        h: frame[(frame.index >= dt.date(2026, 1, 1))]
        for h, frame in per_hour_all.items()
    }
    if len(stub[0]):
        kalman.predict(stub, update_state=True)

    last_temp_smo = float(df["temp_smo"].iloc[-1])
    last_kappa = 0.98  # cf. config.yaml § thermal.kappa (value frozen at training time)

    # --- 3. Fill up to day G with fresh regional data ---
    fresh_start = last_dataset_date + dt.timedelta(days=1)
    try:
        fresh_gas = fetch_fresh_gas_total(fresh_start, day_j)
    except Exception as exc:  # network/ODRÉ unavailable: continue without, state frozen earlier
        warnings.append(f"Failed to fetch fresh (regional) gas consumption: {exc}")
        fresh_gas = pd.Series(dtype=float)

    day_g = last_available_day(fresh_gas) or last_dataset_date
    if day_g <= last_dataset_date:
        warnings.append("No gas data fresher than the existing dataset — Kalman state not refreshed.")

    # --- 4. Weather: actuals (fresh_start -> today) + forecast (today -> J+1) ---
    past_days = max((now.date() - fresh_start).days + 2, 2)
    temp_national = fetch_national_temperature(past_days=past_days, forecast_days=2)
    temp_smo = continue_temp_smo(last_temp_smo, temp_national, last_kappa)

    weather_df = pd.DataFrame({"temp_raw_c": temp_national, "temp_smo": temp_smo})
    weather_df["X1_heating"] = compute_x1_heating(weather_df["temp_raw_c"], 15.0)
    weather_df["X2_smo_heating"] = compute_x2_smo_heating(weather_df["temp_smo"], 15.0)

    calendar_df = _build_calendar_features(weather_df.index)
    features_df = pd.concat([weather_df[["temp_smo", "X1_heating", "X2_smo_heating"]], calendar_df], axis=1)

    # --- 5. Assimilate up to day G (if ground truth is available in this window) ---
    gap_end = pd.Timestamp(day_g, tz="UTC") + pd.Timedelta(hours=23)
    gap_features = features_df.loc[features_df.index <= gap_end].copy()
    if not fresh_gas.empty:
        gap_features[TARGET_COLUMN] = fresh_gas.reindex(gap_features.index)
        gap_features = gap_features.dropna(subset=[TARGET_COLUMN])
        if len(gap_features):
            gap_per_hour = _to_per_hour(gap_features, target_col=TARGET_COLUMN)
            kalman.predict(gap_per_hour, update_state=True)

    # --- 6. Final forecast: J[17h-23h] + J+1[0h-23h], state frozen at day G ---
    horizon_start = pd.Timestamp(day_j, tz="UTC") + pd.Timedelta(hours=17)
    horizon_end = pd.Timestamp(day_j1, tz="UTC") + pd.Timedelta(hours=23)
    horizon_features = features_df.loc[(features_df.index >= horizon_start) & (features_df.index <= horizon_end)]

    if horizon_features.empty:
        warnings.append("No weather data available over the requested horizon (J 17:00 -> J+1 23:00).")

    horizon_per_hour = _to_per_hour(horizon_features)
    kalman_preds = _direct_kalman_predict(kalman, horizon_per_hour)
    ols_beta = ols.beta_
    sure_beta = {h: sure.beta_[h] for h in range(24)}
    ols_preds = _static_predict(ols_beta, PREDICTOR_COLUMNS, horizon_per_hour)
    sure_preds = _static_predict(sure_beta, PREDICTOR_COLUMNS, horizon_per_hour)

    rows = []
    for h in range(24):
        for ts in horizon_per_hour[h]["utc_ts"] if len(horizon_per_hour[h]) else []:
            rows.append({
                "timestamp": ts, "local_hour": h,
                "kalman": kalman_preds[h].get(ts), "ols": ols_preds[h].get(ts), "sure": sure_preds[h].get(ts),
                "weather_source": "forecast" if ts >= pd.Timestamp(now.astimezone(dt.timezone.utc)) else "observed",
            })
    horizon = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    return RealForecastResult(
        day_j=day_j, day_g=day_g, generated_at=now, horizon=horizon, warnings=warnings,
    )
