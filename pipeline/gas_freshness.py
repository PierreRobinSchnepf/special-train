"""Reconstruction of a national gas total fresher than the official aggregate.

Finding (research documented in the pipeline README): the ODRÉ dataset used
for training (`consommation-quotidienne-brute`) publishes its "Définitif"
values with a ~45-50 day lag — unusable for daily assimilation. Two regional
datasets, summed over the regions and the 2 operators (NaTran + Teréga),
reconstruct the same total to within ~0.2%, with the lag reduced to ~15-20
days:
  - `conso-journa-industriel-grtgazterega`      (industrial customers)
  - `courbe-de-charge-eldgrd-regional-grtgaz-terega` (public distribution, DSO/LDC)

A point documented nowhere else and verified empirically (hour-by-hour
comparison against our reference UTC dataset): the hourly columns of these
two datasets are in LOCAL Europe/Paris time (not UTC, and no "gas day"
offset despite the 06h→05h export order of the industrial dataset — that
order is a display artefact, each column stays labeled with its true local
hour). Local→UTC conversion is mandatory before any merge with the rest of
the pipeline (which is all UTC).
"""
from __future__ import annotations

import datetime as dt
import re
from io import StringIO

import pandas as pd
import requests

INDUSTRIAL_DATASET = "conso-journa-industriel-grtgazterega"
DISTRIBUTION_DATASET = "courbe-de-charge-eldgrd-regional-grtgaz-terega"
CALENDAR_TZ = "Europe/Paris"

_HOUR_COL_RE = re.compile(r"^(\d{2})_00")

_SESSION = requests.Session()


def _fetch_export_csv(dataset_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    url = f"https://odre.opendatasoft.com/api/v2/catalog/datasets/{dataset_id}/exports/csv"
    params = {
        "where": f"date in [date'{start.isoformat()}'..date'{end.isoformat()}']",
        "timezone": "UTC",
    }
    resp = _SESSION.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), sep=";")


def _hour_columns(df: pd.DataFrame) -> dict[str, int]:
    """Map each hourly column to its local hour (0-23), whatever the exact
    name (the two datasets use different and sometimes internally
    inconsistent naming conventions — see the module docstring)."""
    mapping = {}
    for col in df.columns:
        m = _HOUR_COL_RE.match(col)
        if m:
            mapping[col] = int(m.group(1))
    return mapping


def _wide_to_utc_series(df: pd.DataFrame) -> pd.Series:
    """Sum all regions/operators, convert local time -> UTC, return a series
    indexed by UTC timestamp (MWh -> treated as MW, consistent with the rest
    of the pipeline: hourly value = average power)."""
    hour_cols = _hour_columns(df)
    if not hour_cols:
        return pd.Series(dtype=float)

    long_rows = []
    dates = pd.to_datetime(df["date"]).dt.date
    for col, hour in hour_cols.items():
        local_ts = pd.to_datetime(dates.astype(str)) + pd.to_timedelta(hour, unit="h")
        long_rows.append(pd.DataFrame({"local_ts": local_ts, "value": df[col]}))

    long_df = pd.concat(long_rows, ignore_index=True).dropna(subset=["value"])
    grouped_local = long_df.groupby("local_ts")["value"].sum()

    # Europe/Paris localization -> UTC. The few ambiguous/nonexistent hours
    # (DST transitions) are dropped rather than guessed.
    localized = grouped_local.index.tz_localize(CALENDAR_TZ, ambiguous="NaT", nonexistent="NaT")
    utc_series = pd.Series(grouped_local.to_numpy(), index=localized).dropna(how="all")
    utc_series = utc_series[utc_series.index.notna()]
    utc_series.index = utc_series.index.tz_convert("UTC")
    return utc_series.sort_index()


def fetch_fresh_gas_total(start: dt.date, end: dt.date) -> pd.Series:
    """Reconstructed national total (industrial + distribution), in MW, UTC-
    indexed, for the [start, end] window. May be shorter than requested on the
    right side when data is not yet published up to `end` — which is exactly
    what detects "day G" (the last available data)."""
    industrial = _fetch_export_csv(INDUSTRIAL_DATASET, start, end)
    distribution = _fetch_export_csv(DISTRIBUTION_DATASET, start, end)

    industrial_utc = _wide_to_utc_series(industrial)
    distribution_utc = _wide_to_utc_series(distribution)

    total = industrial_utc.add(distribution_utc, fill_value=None)
    # add(fill_value=None) keeps NaN when either side is missing at a given
    # hour (no silent partial totals); in practice both sources cover the
    # same days, so this is rare.
    return total.rename("y_gas_mw_fresh").dropna().sort_index()


def last_available_day(series: pd.Series) -> dt.date | None:
    """Last CIVIL day (Europe/Paris) fully covered by `series` (24 hours
    present) — "day G", up to which the state can be updated."""
    if series.empty:
        return None
    local_dates = series.index.tz_convert(CALENDAR_TZ).date
    counts = pd.Series(1, index=local_dates).groupby(level=0).sum()
    complete_days = counts[counts >= 24].index
    return max(complete_days) if len(complete_days) else None
