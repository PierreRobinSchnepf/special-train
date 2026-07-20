"""Lightweight persistence (SQLite) of the forecasts produced by the live
pipeline, so they can later be compared with actual values (once published)
to build a genuine performance history — not a replayed backtest, but real
forecasts made blind about the future at run time.
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
    local_hour INTEGER NOT NULL,
    kalman REAL,
    ols REAL,
    sure REAL,
    weather_source TEXT,
    PRIMARY KEY (run_id, timestamp)
);
"""

# v1.0 stored French column names; renamed in v1.1. Applied once per
# connection open, harmless afterwards (no-op when columns are already new).
_MIGRATIONS = (
    ("heure_locale", "local_hour"),
    ("source_meteo", "weather_source"),
)


def _migrate_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
    for old, new in _MIGRATIONS:
        if old in cols and new not in cols:
            conn.execute(f"ALTER TABLE forecasts RENAME COLUMN {old} TO {new}")
    # v1.0 also stored French values in weather_source.
    conn.execute("UPDATE forecasts SET weather_source='forecast' WHERE weather_source='prevue'")
    conn.execute("UPDATE forecasts SET weather_source='observed' WHERE weather_source='observee'")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    _migrate_columns(conn)
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
        df[["run_id", "generated_at", "day_j", "day_g", "timestamp", "local_hour", "kalman", "ols", "sure", "weather_source"]].to_sql(
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
    """Join recorded forecasts with the actual values available today (fresh
    regional reconstruction, possibly still incomplete for very recent runs —
    see the `pending` column)."""
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
    forecasts["actual"] = actual.reindex(forecasts.index)
    forecasts["pending"] = forecasts["actual"].isna()
    for model in ("kalman", "ols", "sure"):
        forecasts[f"abs_error_{model}"] = (forecasts[model] - forecasts["actual"]).abs()
    return forecasts.reset_index()
