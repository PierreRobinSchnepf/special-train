"""Sauvegarde/chargement des modèles entraînés, pour éviter de tout
ré-entraîner (~40s : OLS + SURE + Kalman) à chaque lancement du dashboard.

Deux jeux de modèles, avec des fenêtres d'entraînement différentes :
  - "backtest"   : utilisé par les onglets Forecast/Benchmark/Suivi. OLS/SURE
    entraînés sur [2020-01-01, 2025-01-01), Kalman sur [2018-01-01, 2025-01-01)
    puis état avancé sur le test 2025 — le test 2025 reste réservé.
  - "production" : utilisé par l'onglet Pipeline réel. Même fenêtre basse,
    mais entraîné jusqu'à fin 2025 inclus (pas de test réservé) — c'est la
    "dernière ré-entraînation avec les données 2025" demandée.

`scripts/train_models.py` construit ces artefacts ; `dashboard/services/
model_store.py` et `pipeline/real_forecast.py` les chargent, et ne
retombent sur un ré-entraînement à la volée que si le cache est absent.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "data" / "models"

OLS_TRAIN_START = "2020-01-01"     # OLS/SURE : on exclut 2018-2019 (trop ancien)
SURE_TRAIN_START = "2020-01-01"
KALMAN_TRAIN_START = "2018-01-01"  # Kalman : historique complet, inchangé

BACKTEST_TEST_START = "2025-01-01"
BACKTEST_TEST_END = "2026-01-01"

# --- Modèles régionaux (Étape C/D) -----------------------------------------
# Un jeu par région et par usage. Nom de fichier :
#   data/models/regional/region<code>_<model>_<set>.pkl
# set ∈ {backtest (hold-out réservé, pour les métriques/dashboard de rejeu),
#        production (entraîné jusqu'à la dernière donnée, pour la prévision)}.
REGIONAL_SUBDIR = "regional"


def regional_artifact_name(region_code: int, model: str, artifact_set: str = "backtest") -> str:
    return f"{REGIONAL_SUBDIR}/region{region_code}_{model}_{artifact_set}"


def _strip_statsmodels_results(obj: Any) -> None:
    """Retire les données d'entraînement (exog/endog) des objets
    statsmodels.RegressionResults enfouis dans nos modèles avant sauvegarde.
    Sans ça, chaque .pkl embarque la matrice de design complète — jusqu'à
    ~700 Mo pour le système SURE (whitened_result_ porte une matrice
    (24*T, 24*k)). Vérifié : `.predict(X)` avec un X explicite (notre seul
    usage post-entraînement) donne un résultat identique avant/après
    `remove_data()` ; seul `.predict()` sans argument casse (jamais utilisé
    ici, on passe toujours un `X` explicite)."""
    # Imports différés : évite un cycle (models/__init__ importe persistence
    # indirectement via dataset.py, qui est importé par ols/sure/kalman).
    from models.kalman import HourlyKalmanSURModel
    from models.ols import HourlyOLSModel
    from models.sure import HourlySUREModel

    if isinstance(obj, HourlyOLSModel):
        for res in obj.results_.values():
            res.remove_data()
    elif isinstance(obj, HourlySUREModel):
        for res in obj.stage1_results_.values():
            res.remove_data()
        if obj.whitened_result_ is not None:
            obj.whitened_result_.remove_data()
    elif isinstance(obj, HourlyKalmanSURModel):
        if obj.sure_ is not None:
            _strip_statsmodels_results(obj.sure_)


def save_artifact(obj: Any, name: str) -> Path:
    _strip_statsmodels_results(obj)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_artifact(name: str) -> Any | None:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
