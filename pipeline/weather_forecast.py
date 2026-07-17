"""Température nationale (actuals + prévision courte échéance) via Open-Meteo.

Pourquoi Open-Meteo plutôt que l'API officielle Météo-France : l'API
officielle (portail-api.meteofrance.fr) demande une inscription manuelle
(email + validation) qu'un agent ne peut pas faire à la place de
l'utilisateur. Open-Meteo est gratuit, sans clé, et sert justement les
prévisions du modèle Météo-France (AROME) pour la France
(`/v1/meteofrance`) — testé et validé pour ce projet.

Un seul appel (`past_days` + `forecast_days`) couvre à la fois :
- les actuals récents, pour combler le trou entre le dernier jour connu de
  conso gaz ("jour G", cf. `pipeline.gas_freshness`) et aujourd'hui — ces
  heures sont déjà arrivées, donc c'est de la vraie donnée observée, pas
  une prévision ;
- la prévision proprement dite, sur le seul horizon qui en a vraiment
  besoin (aujourd'hui restant + J+1), largement dans la plage fiable
  d'une prévision courte échéance.

Les stations et leurs poids sont les mêmes que l'entraînement
(`config.yaml § meteo.stations`) — la seule différence est la source (API
de prévision vs archive climatologique), pas le panier de stations ni la
pondération.
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
    """Température nationale pondérée (même pondération que l'entraînement),
    indexée UTC, de `-past_days` à `+forecast_days` autour d'aujourd'hui."""
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
    if isinstance(payload, dict):  # une seule station demandée -> pas de liste
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
    """Poursuit la récursion EWMA (cf. `src.thermal_features.compute_temp_smo`)
    à partir d'un état existant plutôt que de la réinitialiser à la première
    valeur de `new_temp_raw` — nécessaire pour raccorder la prévision au
    dernier `temp_smo` connu du dataset d'entraînement, sans discontinuité."""
    values = new_temp_raw.to_numpy(dtype=float)
    out = np.empty_like(values)
    prev = seed_temp_smo
    for i, t in enumerate(values):
        prev = kappa * prev + (1 - kappa) * t
        out[i] = prev
    return pd.Series(out, index=new_temp_raw.index, name="temp_smo")
