"""Ingestion and reconstruction of REGIONAL gas consumption (target of the
region-by-region model).

The national dataset `consommation-quotidienne-brute` has no regional
breakdown. Per-region consumption is reconstructed by summing, per region,
two regional ODRÉ datasets:
  - `conso-journa-industriel-grtgazterega`          (industrial customers)
  - `courbe-de-charge-eldgrd-regional-grtgaz-terega` (public distribution, DSO/LDC)

This is the same pair as `pipeline/gas_freshness.py`, with one difference:
this module **keeps the regional granularity** (`code_region`) instead of
summing everything into a national total. The parsing logic is identical and
shares the same empirically verified properties:

- WIDE format: one column per hour (`06_00`, `07_00`, ...; the two datasets
  use inconsistent internal naming conventions — `00_00_00` vs `07_00` —
  hence the tolerant regex `^(\\d{2})_00`);
- **LOCAL Europe/Paris time** (not UTC, no "gas day" offset): local -> UTC
  conversion is mandatory before any merge with the rest of the pipeline
  (which is all UTC);
- several rows per (date, region): one per operator (NaTran + Teréga) and per
  activity sector — **all these rows are summed** to get the regional total.

Always index on the numeric `code_region` (stable), never on the text label
(which varies between the two datasets: "Grand-Est"/"Grand Est",
"Ile-de-France"/"Île-de-France"...).
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.http_utils import download

logger = logging.getLogger(__name__)

CALENDAR_TZ = "Europe/Paris"
_HOUR_COL_RE = re.compile(r"^(\d{2})_00")


# ---------------------------------------------------------------------------
# Step 1: raw ingestion (idempotent cache, per dataset and per year)
# ---------------------------------------------------------------------------

def _raw_dir(config: dict) -> Path:
    return resolve_path(config["output"]["raw_dir"]) / config["gas_regional"]["raw_subdir"]


def _export_url(config: dict, dataset_id: str) -> str:
    return config["gas_regional"]["export_url_template"].format(dataset_id=dataset_id)


def latest_available_source_day(config: dict) -> dt.date | None:
    """Query ODRÉ for the latest day published in the regional SOURCE (both
    datasets update every ~15-20 days). Returns the min of the two maxima (a
    complete total needs both). Used by the "refresh" button to know whether
    there is anything new to download without re-fetching everything."""
    import requests

    reg_cfg = config["gas_regional"]
    date_field = reg_cfg["date_field"]
    maxima = []
    for key in ("industrial_dataset_id", "distribution_dataset_id"):
        url = f"https://odre.opendatasoft.com/api/v2/catalog/datasets/{reg_cfg[key]}/records"
        params = {"select": date_field, "order_by": f"{date_field} desc", "limit": 1}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            recs = resp.json().get("records", [])
            if recs:
                d = recs[0]["record"]["fields"][date_field]
                maxima.append(dt.date.fromisoformat(d[:10]))
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("freshness check failed for %s: %s", key, exc)
    return min(maxima) if maxima else None


def fetch_regional_gas(
    config: dict,
    force: bool = False,
    refresh_current_year: bool = True,
) -> None:
    """Download both regional datasets in yearly chunks into
    data/raw/<raw_subdir>/. Idempotent (skips cached years), except the current
    year which is re-downloaded (data still provisional), like fetch_gas.
    """
    reg_cfg = config["gas_regional"]
    raw_dir = _raw_dir(config)
    date_field = reg_cfg["date_field"]
    request_timeout = reg_cfg.get("request_timeout", 60)

    start_year = dt.datetime.fromisoformat(
        config["date_range"]["start"].replace("Z", "+00:00")
    ).year
    end = config["date_range"]["end"]
    end_year = (
        dt.datetime.fromisoformat(end.replace("Z", "+00:00")).year
        if end
        else dt.datetime.now(dt.timezone.utc).year
    )
    current_year = dt.datetime.now(dt.timezone.utc).year

    datasets = {
        "industrial": reg_cfg["industrial_dataset_id"],
        "distribution": reg_cfg["distribution_dataset_id"],
    }
    for label, dataset_id in datasets.items():
        for year in range(start_year, end_year + 1):
            dest = raw_dir / f"{label}_{year}.csv"
            year_force = force or (refresh_current_year and year == current_year)
            where = f"{date_field} in [date'{year}-01-01'..date'{year}-12-31']"
            params = {"where": where, "timezone": "UTC"}
            logger.info("fetching regional gas %s year=%d (force=%s)", label, year, year_force)
            download(
                _export_url(config, dataset_id), dest,
                params=params, force=year_force, timeout=request_timeout,
            )


# ---------------------------------------------------------------------------
# Step 2: wide (local time) -> long UTC parsing, per region
# ---------------------------------------------------------------------------

def _hour_columns(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(m.group(1)) for col in df.columns if (m := _HOUR_COL_RE.match(col))}


def _wide_to_regional_utc(df: pd.DataFrame, region_code_field: str) -> pd.DataFrame:
    """Convert a WIDE dataframe (local time) into a dataframe indexed by UTC
    timestamp, one column per `code_region`, value = sum over operators and
    sectors. Ambiguous/nonexistent hours (DST transitions) are dropped rather
    than guessed.
    """
    hour_cols = _hour_columns(df)
    if not hour_cols or region_code_field not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df["date"]).dt.normalize()
    codes = pd.to_numeric(df[region_code_field], errors="coerce")

    long_parts = []
    for col, hour in hour_cols.items():
        local_ts = dates + pd.to_timedelta(hour, unit="h")
        long_parts.append(
            pd.DataFrame({"local_ts": local_ts, "code_region": codes, "value": df[col]})
        )
    long_df = pd.concat(long_parts, ignore_index=True).dropna(subset=["value", "code_region"])
    long_df["code_region"] = long_df["code_region"].astype(int)

    # Sum over operators + sectors, per (region, local hour).
    grouped = long_df.groupby(["local_ts", "code_region"])["value"].sum().unstack("code_region")

    # Europe/Paris local time -> UTC.
    localized_idx = grouped.index.tz_localize(
        CALENDAR_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    grouped = grouped[localized_idx.notna()]
    grouped.index = localized_idx[localized_idx.notna()].tz_convert("UTC")
    return grouped.sort_index()


def _load_one_dataset(config: dict, label: str) -> pd.DataFrame:
    """Load and concatenate every cached yearly CSV of one regional dataset;
    return a dataframe UTC (index) x code_region (columns)."""
    reg_cfg = config["gas_regional"]
    raw_dir = _raw_dir(config)
    files = sorted(raw_dir.glob(f"{label}_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no regional gas files '{label}_*.csv' in {raw_dir} — run fetch_regional_gas first"
        )

    per_year = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="utf-8", encoding_errors="replace")
        wide = _wide_to_regional_utc(df, reg_cfg["region_code_field"])
        if not wide.empty:
            per_year.append(wide)

    if not per_year:
        return pd.DataFrame()
    combined = pd.concat(per_year)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _mask_isolated_dips(df: pd.DataFrame, dip_fraction: float) -> pd.DataFrame:
    """Replace with NaN any value < dip_fraction x min(previous neighbor, next
    neighbor): a one-hour collapse surrounded by normal values is physically
    impossible for gas consumption (thermal inertia), hence a source artefact
    (cf. the March DST glitch). Genuine troughs (surrounded by equally low
    hours) are untouched. An edge value (missing one of its two neighbors) is
    never masked (NaN comparison -> False)."""
    prev = df.shift(1)
    nxt = df.shift(-1)
    neighbor_min = np.minimum(prev, nxt)   # NaN if a neighbor is missing
    dip = df < (dip_fraction * neighbor_min)
    n = int(dip.to_numpy().sum())
    if n:
        logger.info("regional gas: masked %d isolated-dip value(s) as NaN (DST-boundary artefact)", n)
    return df.mask(dip)


def load_regional_gas(config: dict) -> pd.DataFrame:
    """Regional target: total consumption (industrial + distribution) per
    region, hourly in UTC. Returns a DataFrame indexed by UTC timestamp, one
    column per `code_region` (int), in MW.

    industrial + distribution are added per (region, hour). A value stays NaN
    when either source is missing at that hour for that region (no silent
    partial totals) — in practice distribution only starts at
    `distribution_start`, so earlier hours are NaN everywhere.
    """
    industrial = _load_one_dataset(config, "industrial")
    distribution = _load_one_dataset(config, "distribution")

    # Union of indexes and columns (regions), then strict sum: NaN whenever
    # either side is missing (add without fill_value propagates NaN).
    total = industrial.add(distribution)

    regions_cfg = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
    known = [c for c in total.columns if c in regions_cfg]
    unexpected = [c for c in total.columns if c not in regions_cfg]
    if unexpected:
        logger.warning("regional gas: unexpected region codes ignored: %s", unexpected)
    total = total[sorted(known)]

    # Cleaning: a regional gas consumption <= 0 is impossible (source gap),
    # then filter one-hour isolated dips (recurring DST artefact).
    total = total.mask(total <= 0)
    total = _mask_isolated_dips(total, float(config["gas_regional"].get("dip_fraction", 0.5)))

    n_regions = total.shape[1]
    logger.info(
        "regional gas: %d regions, %s -> %s (%d hourly points)",
        n_regions, total.index.min(), total.index.max(), len(total),
    )
    return total
