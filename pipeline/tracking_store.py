"""Persistance légère (SQLite) des prévisions produites par le pipeline réel,
pour pouvoir les comparer plus tard aux valeurs réelles (une fois publiées)
et construire un vrai historique de performance — pas un backtest rejoué,
de vraies prévisions faites à l'aveugle sur le futur au moment du run.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from pipeline.real_forecast import RealForecastResult

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "processed" / "tracking.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    run_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    day_j TEXT NOT NULL,
    day_g TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    heure_locale INTEGER NOT NULL,
    kalman REAL,
    ols REAL,
    sure REAL,
    source_meteo TEXT,
    PRIMARY KEY (run_id, timestamp)
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def save_forecast(result: RealForecastResult) -> str:
    run_id = f"{result.day_j.isoformat()}_{uuid.uuid4().hex[:8]}"
    df = result.horizon.copy()
    df["run_id"] = run_id
    df["generated_at"] = result.generated_at.isoformat()
    df["day_j"] = result.day_j.isoformat()
    df["day_g"] = result.day_g.isoformat()
    df["timestamp"] = df["timestamp"].astype(str)

    with _connect() as conn:
        df[["run_id", "generated_at", "day_j", "day_g", "timestamp", "heure_locale", "kalman", "ols", "sure", "source_meteo"]].to_sql(
            "forecasts", conn, if_exists="append", index=False
        )
    return run_id


def load_all_forecasts() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM forecasts", conn)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["generated_at"] = pd.to_datetime(df["generated_at"], utc=True)
    df["day_j"] = pd.to_datetime(df["day_j"]).dt.date
    df["day_g"] = pd.to_datetime(df["day_g"]).dt.date
    return df


def reconcile_with_actuals() -> pd.DataFrame:
    """Rejoint les prévisions enregistrées avec les valeurs réelles
    aujourd'hui disponibles (reconstruction régionale fraîche, éventuellement
    encore incomplète pour les runs très récents — cf. colonne `a_verifier`)."""
    from pipeline.gas_freshness import fetch_fresh_gas_total

    forecasts = load_all_forecasts()
    if forecasts.empty:
        return forecasts

    start = forecasts["timestamp"].min().date()
    end = forecasts["timestamp"].max().date() + dt.timedelta(days=1)
    try:
        actual = fetch_fresh_gas_total(start, end)
    except Exception:
        actual = pd.Series(dtype=float)

    forecasts = forecasts.set_index("timestamp")
    forecasts["reel"] = actual.reindex(forecasts.index)
    forecasts["a_verifier"] = forecasts["reel"].isna()
    for model in ("kalman", "ols", "sure"):
        forecasts[f"erreur_abs_{model}"] = (forecasts[model] - forecasts["reel"]).abs()
    return forecasts.reset_index()
