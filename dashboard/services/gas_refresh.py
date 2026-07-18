"""Actualisation des données gaz régionales + suivi de qualité dans le temps.

Les deux datasets ODRÉ régionaux se mettent à jour ~tous les 15-20 jours. Ce
module :
  1. vérifie s'il existe des données plus récentes en ligne que celles déjà
     construites localement (`check_freshness`) ;
  2. si oui, ré-ingère, reconstruit le dataset régional et ré-entraîne les
     modèles backtest (`run_refresh`) — c'est-à-dire "réentraîne en rajoutant la
     dernière performance" ;
  3. journalise à chaque actualisation la qualité courante des modèles
     (`append_tracking` / `load_tracking`) — un historique MAPE par région dans
     le temps, le "tracking de qualité de prédiction".

Le ré-entraînement est volontairement synchrone (l'appelant l'enveloppe dans un
`st.status`) : ingestion + reconstruction météo + fit ~ quelques minutes.
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
    """Dernier jour couvert par le dataset régional déjà construit (via une
    région de référence). None si le dataset n'existe pas encore."""
    ref = _regional_dir(config) / "dataset_region_11.parquet"
    if not ref.exists():
        return None
    y = pd.read_parquet(ref, columns=["y_gas_mw"])["y_gas_mw"].dropna()
    return y.index.max().date() if not y.empty else None


def check_freshness(config: dict) -> dict:
    """Compare la dernière donnée en ligne au dernier jour construit localement."""
    source_day = latest_available_source_day(config)
    local_day = dataset_last_day(config)
    has_new = bool(source_day and local_day and source_day > local_day)
    gap = (source_day - local_day).days if (source_day and local_day) else None
    return {"source_day": source_day, "local_day": local_day, "has_new": has_new, "gap_days": gap}


def run_refresh(config: dict, log: Callable[[str], None]) -> dict:
    """Ré-ingère, reconstruit et ré-entraîne (jeu backtest). Retourne un résumé.
    `log` reçoit un message par étape (branché sur st.status côté dashboard)."""
    # Imports différés : évitent de charger tout le pipeline au démarrage du dashboard.
    import build_regional_dataset
    import train_regional_models
    from src.regional_gas import fetch_regional_gas

    before = dataset_last_day(config)

    log("Téléchargement des dernières données ODRÉ (industriel + distribution)…")
    fetch_regional_gas(config, refresh_current_year=True)

    log("Reconstruction du dataset régional (12 régions)…")
    build_regional_dataset.main([])

    log("Ré-entraînement des modèles backtest (SURE + Kalman) par région…")
    train_regional_models.main(["--set", "backtest"])

    after = dataset_last_day(config)
    log("Mise à jour du suivi de qualité…")
    entry = append_tracking(config, before, after)
    log("Terminé.")
    return {"before": before, "after": after, "tracking_entry": entry}


# ---------------------------------------------------------------------------
# Suivi de qualité dans le temps (un point par actualisation)
# ---------------------------------------------------------------------------

def _tracking_path(config: dict) -> Path:
    return _regional_dir(config) / "refresh_tracking.json"


def _current_quality(config: dict) -> dict:
    """MAPE Kalman par région (jeu backtest) tel qu'il vient d'être recalculé."""
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
