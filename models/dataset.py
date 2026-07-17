"""Préparation du panel jour x heure utilisé par les deux modèles benchmark.

Les deux modèles (OLS et SURE) reposent sur la même décomposition en 24
équations horaires : pour chaque heure locale h ∈ [0,23], une équation prédit
`y_gas_mw` à partir des prédicteurs du Tableau 1 évalués à cette heure-là.
Le panel est rendu équilibré (mêmes jours pour les 24 équations) car
l'estimation SURE a besoin d'observations contemporaines alignées pour
calculer la covariance inter-équations — les rares jours de transition
DST (heure locale manquante ou dupliquée, ~2/an) sont exclus plutôt que
bricolés.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "dataset_final.parquet"
CALENDAR_TZ = "Europe/Paris"

TARGET_COLUMN = "y_gas_mw"

# Prédicteurs du Tableau 1 uniquement (temp_raw_c est exclue : c'est une
# variable brute intermédiaire conservée pour audit, pas une variable du
# modèle — voir data_dictionary.md).
_THERMAL = ["temp_smo", "X1_heating", "X2_smo_heating"]
_FOURIER = [
    f"{trig}{s}_{grp}"
    for trig in ("cos", "sin")
    for s in (1, 2, 3, 4)
    for grp in ("WD", "WE")
]
_CALENDAR = ["is_monday", "is_friday", "is_saturday", "is_sunday", "is_end_of_year", "is_off_peak_period"]
_INTERCEPT = ["beta_0"]

PREDICTOR_COLUMNS = _THERMAL + _FOURIER + _CALENDAR + _INTERCEPT


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_hourly_equations(
    df: pd.DataFrame,
    predictor_cols: list[str] = PREDICTOR_COLUMNS,
    target_col: str = TARGET_COLUMN,
) -> dict[int, pd.DataFrame]:
    """Découpe `df` (index horaire UTC) en 24 DataFrames (un par heure locale),
    indexés par date locale, et équilibrés (mêmes dates pour les 24 heures).

    Chaque DataFrame retourné a pour colonnes [target_col, *predictor_cols,
    "utc_ts"] — `utc_ts` porte l'horodatage UTC d'origine, nécessaire pour
    replacer les prédictions sur l'index temporel complet sans avoir à
    reconstruire une date locale (ambiguë aux transitions DST).
    """
    local = df.index.tz_convert(CALENDAR_TZ)
    work = df[[target_col, *predictor_cols]].copy()
    work["local_date"] = local.date
    work["local_hour"] = local.hour
    work["utc_ts"] = df.index

    work = work.dropna(subset=[target_col, *predictor_cols])
    # Jour de bascule automne (heure locale dupliquée) : on garde la première
    # occurrence (avant le changement d'heure) par convention documentée.
    work = work.drop_duplicates(subset=["local_date", "local_hour"], keep="first")

    per_hour = {
        h: work.loc[work["local_hour"] == h].set_index("local_date").sort_index()
        for h in range(24)
    }

    common_dates = set(per_hour[0].index)
    for h in range(1, 24):
        common_dates &= set(per_hour[h].index)
    common_dates = sorted(common_dates)

    n_before = {h: len(per_hour[h]) for h in range(24)}
    per_hour = {h: per_hour[h].loc[common_dates] for h in range(24)}
    n_dropped = {h: n_before[h] - len(common_dates) for h in range(24) if n_before[h] != len(common_dates)}
    if n_dropped:
        total_candidate_days = len(set().union(*(set(per_hour[h].index) for h in range(24))) | set(n_before))
        print(f"build_hourly_equations: panel équilibré sur {len(common_dates)} jours "
              f"(jours exclus par heure pour déséquilibre : {n_dropped})")

    return per_hour


def split_train_test(
    per_hour: dict[int, pd.DataFrame],
    test_start: str = "2025-01-01",
    test_end: str = "2026-01-01",
    train_start: str | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Train = tout ce qui précède `test_start` (et suit `train_start` si
    fourni). Test = [test_start, test_end).

    Les données postérieures à `test_end` (ex. le reliquat partiel de la
    dernière année du dataset) sont volontairement exclues des deux
    ensembles : elles ne constituent ni un historique d'entraînement propre
    (postérieures à la période de test) ni une période de test complète.

    `train_start` permet d'exclure les données trop anciennes (ex. OLS/SURE
    entraînés seulement depuis 2020 — cf. `models/persistence.py`). Passer
    `test_start == test_end` donne un test vide et un train couvrant tout
    l'historique jusqu'à cette date (utile pour les modèles "production" qui
    n'ont pas besoin d'un jeu de test réservé).
    """
    test_start_d = pd.Timestamp(test_start).date()
    test_end_d = pd.Timestamp(test_end).date()
    train_start_d = pd.Timestamp(train_start).date() if train_start is not None else None

    def _train_mask(index: pd.Index) -> pd.Index:
        mask = index < test_start_d
        if train_start_d is not None:
            mask &= index >= train_start_d
        return mask

    train = {h: frame.loc[_train_mask(frame.index)] for h, frame in per_hour.items()}
    test = {h: frame.loc[(frame.index >= test_start_d) & (frame.index < test_end_d)] for h, frame in per_hour.items()}
    return train, test


def target_series(per_hour: dict[int, pd.DataFrame], target_col: str = TARGET_COLUMN) -> dict[int, pd.Series]:
    """Cible indexée par `utc_ts`, dans le même format que les prédictions des modèles."""
    return {h: frame.set_index("utc_ts")[target_col] for h, frame in per_hour.items()}
