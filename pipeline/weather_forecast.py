"""National temperature (actuals + short-range forecast) via Open-Meteo.

Why Open-Meteo rather than the official Météo-France API: the official API
(portail-api.meteofrance.fr) requires a manual registration (email +
validation) that an agent cannot do on the user's behalf. Open-Meteo is free,
keyless, and serves precisely the Météo-France model forecasts (AROME) for
France (`/v1/meteofrance`) — tested and validated for this project.

A single call (`past_days` + `forecast_days`) covers both:
- recent actuals, to fill the gap between the last known gas-consumption day
  ("day G", see `pipeline.gas_freshness`) and today — these hours have
  already happened, so this is genuinely observed data, not a forecast;
- the forecast proper, over the only horizon that really needs one (the rest
  of today + J+1), well within the reliable range of a short-range forecast.

The stations and their weights are the same as in training
(`config.yaml § meteo.stations`) — the only difference is the source
(forecast API vs climatological archive), not the station basket nor the
weighting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
STATION_COORDS_PATH = REPO_ROOT / "dashboard" / "services" / "station_coords.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/meteofrance"


def _load_station_coords() -> dict:
    return json.loads(STATION_COORDS_PATH.read_text(encoding="utf-8"))


def fetch_national_temperature(past_days: int, forecast_days: int = 2) -> pd.Series:
    """Weighted national temperature (same weighting as training), UTC-
    indexed, from `-past_days` to `+forecast_days` around today."""
    coords = _load_station_coords()
    config = load_config()
    weights = {s["department"]: float(s["weight"]) for s in config["meteo"]["stations"]}

    departments = list(coords.keys())
    lats = ",".join(str(coords[d]["lat"]) for d in departments)
    lons = ",".join(str(coords[d]["lon"]) for d in departments)

    params = {
        "latitude": lats, "longitude": lons,
        "hourly": "temperature_2m",
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):  # single station requested -> no list
        payload = [payload]

    per_station = {}
    for dep, station_payload in zip(departments, payload):
        idx = pd.to_datetime(station_payload["hourly"]["time"], utc=True)
        per_station[dep] = pd.Series(station_payload["hourly"]["temperature_2m"], index=idx)

    wide = pd.DataFrame(per_station)
    weight_series = pd.Series(weights)
    row_weights = wide.notna().mul(weight_series, axis=1)
    weighted = (wide.fillna(0) * row_weights).sum(axis=1) / row_weights.sum(axis=1)
    return weighted.rename("temp_raw_c").sort_index()


def continue_temp_smo(seed_temp_smo: float, new_temp_raw: pd.Series, kappa: float) -> pd.Series:
    """Continue the EWMA recursion (see `src.thermal_features.compute_temp_smo`)
    from an existing state rather than reinitializing it at the first value of
    `new_temp_raw` — required to join the forecast onto the last known
    `temp_smo` of the training dataset without a discontinuity."""
    values = new_temp_raw.to_numpy(dtype=float)
    out = np.empty_like(values)
    prev = seed_temp_smo
    for i, t in enumerate(values):
        prev = kappa * prev + (1 - kappa) * t
        out[i] = prev
    return pd.Series(out, index=new_temp_raw.index, name="temp_smo")
