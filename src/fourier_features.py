"""Bloc saisonnier (Fourier) du Tableau 1, avec dédoublement jours ouvrés / week-end.

Pour chaque harmonique s, cos_s/sin_s sont calculés à partir du jour de
l'année civil français (Europe/Paris), puis masqués à 0 selon que l'heure
tombe un jour ouvré (WD) ou un week-end (WE). Chaque timestamp porte une
valeur non nulle dans exactement un des deux jeux de colonnes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fourier_features(
    index: pd.DatetimeIndex,
    harmonics: list[int],
    days_in_year: float,
    calendar_tz: str,
) -> pd.DataFrame:
    if index.tz is None:
        raise ValueError("index must be tz-aware (UTC)")

    local = index.tz_convert(calendar_tz)
    day_of_year = local.dayofyear.to_numpy(dtype=float)
    is_weekend = local.dayofweek.to_numpy() >= 5  # 5=samedi, 6=dimanche

    cols: dict[str, np.ndarray] = {}
    for s in harmonics:
        angle = 2.0 * np.pi * s * day_of_year / days_in_year
        cos_s = np.cos(angle)
        sin_s = np.sin(angle)

        cols[f"cos{s}_WD"] = np.where(~is_weekend, cos_s, 0.0)
        cols[f"sin{s}_WD"] = np.where(~is_weekend, sin_s, 0.0)
        cols[f"cos{s}_WE"] = np.where(is_weekend, cos_s, 0.0)
        cols[f"sin{s}_WE"] = np.where(is_weekend, sin_s, 0.0)

    # Ordre demandé par la spec : cos1..cos4 puis sin1..sin4, pour WD puis WE.
    ordered_columns = (
        [f"cos{s}_WD" for s in harmonics]
        + [f"sin{s}_WD" for s in harmonics]
        + [f"cos{s}_WE" for s in harmonics]
        + [f"sin{s}_WE" for s in harmonics]
    )
    return pd.DataFrame(cols, index=index)[ordered_columns]
