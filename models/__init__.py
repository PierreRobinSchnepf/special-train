"""Modèles de benchmark pour la prévision horaire de consommation de gaz.

Deux approches, toutes deux structurées en 24 équations (une par heure locale
Europe/Paris), avec les mêmes prédicteurs (Tableau 1 du rapport) :

- `models.ols.HourlyOLSModel`   : 24 régressions OLS indépendantes.
- `models.sure.HourlySUREModel` : même système estimé conjointement par FGLS
  (Seemingly Unrelated Regressions, Zellner 1962), qui exploite la
  corrélation contemporaine des résidus entre équations (un même jour).

Voir `models.dataset` pour la préparation du panel équilibré par jour/heure
et le split train/test, et `models.metrics` pour RMSE/MAPE.
"""
from models.dataset import (
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_hourly_equations,
    load_dataset,
    split_train_test,
)
from models.kalman import HourlyKalmanSURModel
from models.ols import HourlyOLSModel
from models.sure import HourlySUREModel

__all__ = [
    "PREDICTOR_COLUMNS",
    "TARGET_COLUMN",
    "load_dataset",
    "build_hourly_equations",
    "split_train_test",
    "HourlyOLSModel",
    "HourlySUREModel",
    "HourlyKalmanSURModel",
]
