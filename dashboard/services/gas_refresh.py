"""Regional gas data refresh + quality tracking over time.

Both regional ODRÉ datasets update every ~15-20 days. This module:
  1. checks whether data newer than what was built locally is available
     online (`check_freshness`);
  2. if so, re-ingests, rebuilds the regional dataset and retrains the
     backtest models (`run_refresh`) — i.e. "retrain including the latest
     performance";
  3. logs the models' current quality at every refresh
     (`append_tracking` / `load_tracking`) — a per-region MAPE history over
     time, the "prediction quality tracking".

The retraining is deliberately synchronous (the caller wraps it in a
`st.status`): ingestion + weather rebuild + fit ~ a few minutes.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import resolve_path
from src.regional_gas import latest_available_source_day


def _regional_dir(config: dict) -> Path:
    subdir = config["regional_models"]["dataset_subdir"]
    return resolve_path(config["output"]["processed_dir"]) / subdir


def dataset_last_day(config: dict) -> dt.date | None:
    """Last day covered by the already-built regional dataset (through a
    reference region). None when the dataset does not exist yet."""
    ref = _regional_dir(config) / "dataset_region_11.parquet"
    if not ref.exists():
        return None
    y = pd.read_parquet(ref, columns=["y_gas_mw"])["y_gas_mw"].dropna()
    return y.index.max().date() if not y.empty else None


def check_freshness(config: dict) -> dict:
    """Compare the latest online data with the last locally built day."""
    source_day = latest_available_source_day(config)
    local_day = dataset_last_day(config)
    has_new = bool(source_day and local_day and source_day > local_day)
    gap = (source_day - local_day).days if (source_day and local_day) else None
    return {"source_day": source_day, "local_day": local_day, "has_new": has_new, "gap_days": gap}


def run_refresh(config: dict, log: Callable[[str], None]) -> dict:
    """Re-ingest, rebuild and retrain (backtest set). Returns a summary.
    `log` receives one message per step (wired to st.status dashboard-side)."""
    # Deferred imports: avoid loading the whole pipeline at dashboard startup.
    from scripts import build_regional_dataset, train_regional_models
    from src.regional_gas import fetch_regional_gas

    before = dataset_last_day(config)

    log("Downloading the latest ODRÉ data (industrial + distribution)…")
    fetch_regional_gas(config, refresh_current_year=True)

    log("Rebuilding the regional dataset (12 regions)…")
    build_regional_dataset.main([])

    log("Retraining the backtest models (SURE + Kalman) per region…")
    train_regional_models.main(["--set", "backtest"])

    after = dataset_last_day(config)
    log("Updating the quality tracking…")
    entry = append_tracking(config, before, after)
    log("Done.")
    return {"before": before, "after": after, "tracking_entry": entry}


# ---------------------------------------------------------------------------
# Quality tracking over time (one point per refresh)
# ---------------------------------------------------------------------------

def _tracking_path(config: dict) -> Path:
    return _regional_dir(config) / "refresh_tracking.json"


def _current_quality(config: dict) -> dict:
    """Kalman MAPE per region (backtest set) as just recomputed."""
    from dashboard.services.map_data import regional_error_table

    tbl = regional_error_table(config)
    per_region = {r["code"]: r["mape"] for _, r in tbl.iterrows()}
    mean_mape = float(tbl["mape"].mean()) if not tbl.empty else None
    return {"mean_mape_kalman": mean_mape, "per_region_mape": per_region}


def append_tracking(config: dict, day_before: dt.date | None, day_after: dt.date | None) -> dict:
    path = _tracking_path(config)
    history = load_tracking(config)
    entry = {
        "refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_last_day_before": day_before.isoformat() if day_before else None,
        "data_last_day_after": day_after.isoformat() if day_after else None,
        **_current_quality(config),
    }
    history.append(entry)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def load_tracking(config: dict) -> list[dict]:
    path = _tracking_path(config)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
