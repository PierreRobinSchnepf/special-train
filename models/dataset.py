"""Preparation of the day x hour panel used by the benchmark models.

Both models (OLS and SURE) rely on the same decomposition into 24 hourly
equations: for each local hour h ∈ [0,23], one equation predicts `y_gas_mw`
from the Table 1 predictors evaluated at that hour. The panel is balanced
(same days across the 24 equations) because the SURE estimation needs aligned
contemporaneous observations to compute the cross-equation covariance — the
few DST-transition days (missing or duplicated local hour, ~2/year) are
excluded rather than patched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "dataset_final.parquet"
CALENDAR_TZ = "Europe/Paris"

TARGET_COLUMN = "y_gas_mw"

# Table 1 predictors only (temp_raw_c is excluded: it is an intermediate raw
# variable kept for audit, not a model variable — see docs/data-dictionary.md).
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
    """Split `df` (UTC hourly index) into 24 DataFrames (one per local hour),
    indexed by local date, and balanced (same dates across the 24 hours).

    Each returned DataFrame has columns [target_col, *predictor_cols,
    "utc_ts"] — `utc_ts` carries the original UTC timestamp, needed to place
    predictions back on the full time index without reconstructing a local
    date (ambiguous at DST transitions).
    """
    local = df.index.tz_convert(CALENDAR_TZ)
    work = df[[target_col, *predictor_cols]].copy()
    work["local_date"] = local.date
    work["local_hour"] = local.hour
    work["utc_ts"] = df.index

    work = work.dropna(subset=[target_col, *predictor_cols])
    # Autumn switch day (duplicated local hour): keep the first occurrence
    # (before the clock change) by documented convention.
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
        print(f"build_hourly_equations: panel balanced over {len(common_dates)} days "
              f"(days dropped per hour for imbalance: {n_dropped})")

    return per_hour


def split_train_test(
    per_hour: dict[int, pd.DataFrame],
    test_start: str = "2025-01-01",
    test_end: str = "2026-01-01",
    train_start: str | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Train = everything before `test_start` (and after `train_start` when
    given). Test = [test_start, test_end).

    Data after `test_end` (e.g. the partial remainder of the dataset's last
    year) is deliberately excluded from both sets: it is neither a clean
    training history (it postdates the test period) nor a complete test
    period.

    `train_start` excludes data that is too old (e.g. OLS/SURE trained only
    from 2020 on — see `models/persistence.py`). Passing
    `test_start == test_end` yields an empty test set and a train set covering
    the whole history up to that date (useful for "production" models that
    need no held-out test set).
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
    """Target indexed by `utc_ts`, in the same format as model predictions."""
    return {h: frame.set_index("utc_ts")[target_col] for h, frame in per_hour.items()}
