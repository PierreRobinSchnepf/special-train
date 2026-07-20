"""Seasonal (Fourier) block of Table 1, split between weekdays and weekends.

For each harmonic s, cos_s/sin_s are computed from the French civil day of
year (Europe/Paris), then masked to 0 depending on whether the hour falls on
a weekday (WD) or a weekend (WE). Each timestamp carries a non-zero value in
exactly one of the two column sets.
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
    is_weekend = local.dayofweek.to_numpy() >= 5  # 5=Saturday, 6=Sunday

    cols: dict[str, np.ndarray] = {}
    for s in harmonics:
        angle = 2.0 * np.pi * s * day_of_year / days_in_year
        cos_s = np.cos(angle)
        sin_s = np.sin(angle)

        cols[f"cos{s}_WD"] = np.where(~is_weekend, cos_s, 0.0)
        cols[f"sin{s}_WD"] = np.where(~is_weekend, sin_s, 0.0)
        cols[f"cos{s}_WE"] = np.where(is_weekend, cos_s, 0.0)
        cols[f"sin{s}_WE"] = np.where(is_weekend, sin_s, 0.0)

    # Column order required by the spec: cos1..cos4 then sin1..sin4, WD then WE.
    ordered_columns = (
        [f"cos{s}_WD" for s in harmonics]
        + [f"sin{s}_WD" for s in harmonics]
        + [f"cos{s}_WE" for s in harmonics]
        + [f"sin{s}_WE" for s in harmonics]
    )
    return pd.DataFrame(cols, index=index)[ordered_columns]
