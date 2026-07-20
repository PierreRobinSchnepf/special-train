"""Data for the regional choropleth map on the home screen.

Reads the GeoJSON base map (committed asset) and the per-region error table
(backtest-set metrics, `data/processed/regional/metrics_regional.json`).
The map colors each region by the Kalman model's **signed bias** (actual vs
predicted) over the test window — the requested "difference between actual
and predicted consumption" indicator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import resolve_path

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
GEOJSON_PATH = _ASSETS / "regions_metropole.geojson"


def load_geojson() -> dict:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        gj = json.load(f)
    # Normalize the join key: featureidkey = properties.code (str).
    for feat in gj["features"]:
        feat["properties"]["code"] = str(feat["properties"]["code"])
    return gj


def _metrics_path(config: dict) -> Path:
    subdir = config["regional_models"]["dataset_subdir"]
    return resolve_path(config["output"]["processed_dir"]) / subdir / "metrics_regional.json"


def regional_error_table(config: dict) -> pd.DataFrame:
    """One record per region: code, label, and the Kalman test metrics
    (signed bias %, MAPE %, RMSE MW). Returns an empty DataFrame when the
    metrics have not been generated yet (train_regional_models.py)."""
    path = _metrics_path(config)
    if not path.exists():
        return pd.DataFrame(columns=["code", "region", "bias_pct", "mape", "rmse"])

    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for code, m in metrics.items():
        if code.startswith("_"):
            continue
        k = m.get("kalman", {}).get("test", {})
        rows.append({
            "code": str(code),
            "region": m.get("region_label", code),
            "bias_pct": k.get("bias_pct"),
            "mape": k.get("mape"),
            "rmse": k.get("rmse"),
        })
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
