"""Données pour la carte choroplèthe régionale de l'écran d'accueil.

Lit le fond de carte GeoJSON (asset committé) et le tableau d'erreur par région
(métriques du jeu backtest, `data/processed/regional/metrics_regional.json`).
La carte colore chaque région par le **biais signé** (réel vs prévu) du modèle
Kalman sur la fenêtre de test — l'indicateur demandé "différence entre
consommation réelle et consommation prédite".
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
    # Homogénéise la clé de jointure : featureidkey = properties.code (str).
    for feat in gj["features"]:
        feat["properties"]["code"] = str(feat["properties"]["code"])
    return gj


def _metrics_path(config: dict) -> Path:
    subdir = config["regional_models"]["dataset_subdir"]
    return resolve_path(config["output"]["processed_dir"]) / subdir / "metrics_regional.json"


def regional_error_table(config: dict) -> pd.DataFrame:
    """Un enregistrement par région : code, libellé, et métriques test du Kalman
    (biais signé %, MAPE %, RMSE MW). Retourne un DataFrame vide si les métriques
    n'ont pas encore été générées (train_regional_models.py)."""
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
