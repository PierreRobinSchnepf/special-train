"""Orchestrateur principal (étapes 1→5) : ingestion, alignement, features, export.

Usage :
    python build_dataset.py                      # run complet 2018-présent
    python build_dataset.py --sample-months 1     # 1 mois pour test rapide
    python build_dataset.py --skip-fetch          # réutilise le cache existant
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.calendar_features import (
    compute_end_of_year_flag,
    compute_off_peak_flag,
    compute_weekday_flags,
)
from src.config import load_config, resolve_path
from src.fourier_features import compute_fourier_features
from src.thermal_features import compute_temp_smo, compute_x1_heating, compute_x2_smo_heating
from fetch_data import fetch_gas, fetch_holidays, fetch_meteo, fetch_school_holidays

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")


# ---------------------------------------------------------------------------
# Chargement des sources brutes
# ---------------------------------------------------------------------------

def load_gas(config: dict) -> pd.Series:
    gas_cfg = config["gas"]
    raw_dir = resolve_path(config["output"]["raw_dir"]) / "gas"
    files = sorted(raw_dir.glob("gas_*.csv"))
    if not files:
        raise FileNotFoundError(f"no gas files found in {raw_dir} — run fetch_data.py first")

    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        frames.append(df)
    gas = pd.concat(frames, ignore_index=True)

    gas[gas_cfg["datetime_field"]] = pd.to_datetime(gas[gas_cfg["datetime_field"]], utc=True)
    gas = gas.drop_duplicates(subset=[gas_cfg["datetime_field"]])
    gas = gas.set_index(gas_cfg["datetime_field"]).sort_index()

    # Le gaz est publié à la maille horaire pleine ; les points :30 sont une
    # structure héritée de la table électricité (demi-horaire) et sont
    # toujours vides pour le gaz — on les élimine explicitement plutôt que
    # de s'appuyer sur le NaN (qui doit rester réservé aux vraies données
    # manquantes sur la maille horaire).
    gas = gas[gas.index.minute == 0]

    series = gas[gas_cfg["value_field"]].rename("y_gas_mw")
    n_missing = series.isna().sum()
    logger.info("gas: %d hourly points loaded, %d missing (%.2f%%)", len(series), n_missing, 100 * n_missing / len(series))
    return series


def _pick_best_station(df: pd.DataFrame, station_field: str, value_field: str) -> str:
    coverage = df.groupby(station_field)[value_field].apply(lambda s: s.notna().sum())
    return coverage.idxmax()


def load_meteo_national(config: dict) -> pd.Series:
    meteo_cfg = config["meteo"]
    raw_dir = resolve_path(config["output"]["raw_dir"]) / "meteo"
    station_field = meteo_cfg["station_id_field"]
    dt_field = meteo_cfg["datetime_field"]
    value_field = meteo_cfg["value_field"]

    per_dept_series: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    selection_log: dict[str, dict] = {}

    for station_cfg in meteo_cfg["stations"]:
        dep = station_cfg["department"]
        files = sorted(raw_dir.glob(f"H_{dep}_*.csv.gz"))
        if not files:
            logger.warning("no meteo files for department %s, excluded from aggregation", dep)
            continue

        frames = []
        for f in files:
            df = pd.read_csv(
                f,
                sep=";",
                usecols=[station_field, dt_field, value_field],
                dtype={station_field: str, dt_field: str},
                compression="gzip",
            )
            frames.append(df)
        dep_df = pd.concat(frames, ignore_index=True)

        best_station = _pick_best_station(dep_df, station_field, value_field)
        dep_df = dep_df[dep_df[station_field] == best_station].copy()
        dep_df[dt_field] = pd.to_datetime(dep_df[dt_field], format=meteo_cfg["datetime_format"], utc=True)
        dep_df = dep_df.drop_duplicates(subset=[dt_field]).set_index(dt_field).sort_index()

        series = dep_df[value_field].rename(f"T_{dep}")
        per_dept_series[dep] = series
        weights[dep] = float(station_cfg["weight"])
        selection_log[dep] = {
            "region_label": station_cfg.get("region_label"),
            "selected_station": best_station,
            "n_stations_in_department": int(dep_df[station_field].nunique()) if station_field in dep_df else None,
            "n_observations": int(series.notna().sum()),
            "coverage_start": str(series.first_valid_index()),
            "coverage_end": str(series.last_valid_index()),
        }
        logger.info("department %s: selected station %s (%d obs, %s -> %s)",
                    dep, best_station, series.notna().sum(),
                    series.first_valid_index(), series.last_valid_index())

    sel_path = raw_dir / "station_selection.json"
    sel_path.write_text(json.dumps(selection_log, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote station selection log to %s", sel_path)

    if not per_dept_series:
        raise RuntimeError("no meteo data loaded for any configured department")

    wide = pd.DataFrame(per_dept_series)

    # Comblement des trous courts (capteur hors ligne quelques heures) avant
    # toute agrégation ou lissage — trous plus longs laissés en NaN.
    max_gap = meteo_cfg["max_ffill_gap_hours"]
    wide_filled = wide.ffill(limit=max_gap)

    weight_series = pd.Series(weights)
    available_mask = wide_filled.notna()
    # Renormalisation des poids par ligne sur les départements disponibles,
    # pour ne pas biaiser la moyenne nationale quand un département manque.
    row_weights = available_mask.mul(weight_series, axis=1)
    row_weight_sums = row_weights.sum(axis=1)
    weighted_values = (wide_filled.fillna(0) * row_weights).sum(axis=1)

    national_temp = weighted_values / row_weight_sums
    national_temp[row_weight_sums == 0] = np.nan
    national_temp = national_temp.rename("temp_c")

    n_partial = (available_mask.sum(axis=1) < len(weights)).sum()
    logger.info(
        "meteo: %d/%d departments aggregated, %d hour(s) used renormalized weights (missing dept), %d fully missing",
        len(weights), len(meteo_cfg["stations"]), n_partial, int((row_weight_sums == 0).sum()),
    )
    return national_temp


def load_holidays(config: dict) -> set[dt.date]:
    path = resolve_path(config["output"]["raw_dir"]) / "holidays" / "jours_feries_metropole.csv"
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"]).dt.date
    return set(dates)


def load_school_holidays(config: dict) -> set[dt.date]:
    sh_cfg = config["school_holidays"]
    path = resolve_path(config["output"]["raw_dir"]) / "school_holidays" / "calendrier_scolaire.csv"
    df = pd.read_csv(path)
    df = df[df["zones"].isin(sh_cfg["zones_metropole"])]
    starts = pd.to_datetime(df["start_date"], utc=True)
    ends = pd.to_datetime(df["end_date"], utc=True)

    calendar_tz = config["timezone"]["calendar_reference"]
    all_dates: set[dt.date] = set()
    for s, e in zip(starts, ends):
        s_local = s.tz_convert(calendar_tz).date()
        e_local = e.tz_convert(calendar_tz).date()
        all_dates.update(pd.date_range(s_local, e_local, freq="D").date)
    return all_dates


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------

def build_master_index(config: dict, gas: pd.Series, temp: pd.Series) -> pd.DatetimeIndex:
    start = pd.Timestamp(config["date_range"]["start"])
    end_cfg = config["date_range"]["end"]
    if end_cfg:
        end = pd.Timestamp(end_cfg)
    else:
        end = min(gas.index.max(), temp.index.max())
    return pd.date_range(start, end, freq="h", tz="UTC")


def assemble(config: dict) -> tuple[pd.DataFrame, dict]:
    gas = load_gas(config)
    temp = load_meteo_national(config)

    index = build_master_index(config, gas, temp)
    logger.info("master index: %s -> %s (%d hourly points)", index.min(), index.max(), len(index))

    gas = gas.reindex(index)
    temp = temp.reindex(index)

    thermal_cfg = config["thermal"]
    x1 = compute_x1_heating(temp, thermal_cfg["t_base_celsius"])
    temp_smo = compute_temp_smo(temp, thermal_cfg["kappa"])
    x2 = compute_x2_smo_heating(temp_smo, thermal_cfg["t_base_celsius"])

    fourier_cfg = config["fourier"]
    calendar_tz = config["timezone"]["calendar_reference"]
    fourier_df = compute_fourier_features(index, fourier_cfg["harmonics"], fourier_cfg["days_in_year"], calendar_tz)

    weekday_flags = compute_weekday_flags(index, calendar_tz)
    end_of_year = compute_end_of_year_flag(index, calendar_tz, config["calendar"]["end_of_year_window"])

    holiday_dates = load_holidays(config)
    school_dates = load_school_holidays(config)
    off_peak = compute_off_peak_flag(index, calendar_tz, holiday_dates, school_dates, end_of_year)

    beta_0 = pd.Series(1, index=index, name="beta_0")

    df = pd.concat(
        [
            gas.rename("y_gas_mw"),
            temp.rename("temp_raw_c"),
            x1,
            temp_smo,
            x2,
            fourier_df,
            weekday_flags,
            end_of_year,
            off_peak,
            beta_0,
        ],
        axis=1,
    )

    qc = run_qc(df, index)
    return df, qc


def run_qc(df: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    nan_pct = (df.isna().mean() * 100).round(3).to_dict()
    n_dupes = int(index.duplicated().sum())
    expected_len = len(pd.date_range(index.min(), index.max(), freq="h"))
    n_gaps = expected_len - len(index)
    qc = {
        "date_range": [str(index.min()), str(index.max())],
        "n_rows": len(df),
        "n_duplicate_timestamps": n_dupes,
        "n_missing_hours_in_index": int(n_gaps),
        "nan_pct_by_column": nan_pct,
    }
    for col, pct in nan_pct.items():
        if pct > 0:
            logger.info("QC: column %-20s %.3f%% NaN", col, pct)
    if n_dupes:
        logger.warning("QC: %d duplicate timestamps found (post-DST check)", n_dupes)
    if n_gaps:
        logger.warning("QC: %d missing hour(s) in the master index", n_gaps)
    return qc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fetch", action="store_true", help="réutilise le cache data/raw/ existant")
    parser.add_argument("--force-fetch", action="store_true", help="force le retéléchargement de tout")
    parser.add_argument("--sample-months", type=int, default=None, help="restreint la fenêtre de dates pour un test rapide")
    args = parser.parse_args(argv)

    config = load_config()

    if args.sample_months is not None:
        start = pd.Timestamp(config["date_range"]["start"])
        end = start + pd.DateOffset(months=args.sample_months)
        config["date_range"]["end"] = end.isoformat()
        logger.info("sample mode: restricting date range to %s -> %s", start, end)

    if not args.skip_fetch:
        fetch_gas(config, force=args.force_fetch)
        fetch_meteo(config, force=args.force_fetch)
        fetch_holidays(config, force=args.force_fetch)
        fetch_school_holidays(config, force=args.force_fetch)

    df, qc = assemble(config)

    processed_dir = resolve_path(config["output"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = resolve_path(config["output"]["parquet_path"])
    df.to_parquet(parquet_path)
    logger.info("wrote %s (%d rows, %d columns)", parquet_path, *df.shape)

    qc_path = processed_dir / "qc_report.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote QC report to %s", qc_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
