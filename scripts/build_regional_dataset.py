"""Build of the REGIONAL dataset (Step B/C of the region-by-region model).

Produces, for each of the 12 gas regions, a DataFrame with the **same schema
as the national dataset** (`dataset_final.parquet`): target `y_gas_mw` +
Table 1 predictors. The existing models (`models/ols.py`, `models/sure.py`,
`models/kalman.py`) and `models/dataset.build_hourly_equations` therefore
apply as-is, region by region, without any modification.

What is SPECIFIC to each region:
  - `y_gas_mw`       : regional consumption (industrial + distribution), UTC;
  - `temp_raw_c`     : regional temperature (population-weighted average of
                        the region's stations);
  - `temp_smo`, `X1_heating`, `X2_smo_heating`: thermal block recomputed on
                        the region's temperature.
What is SHARED (national, identical for every region):
  - Fourier (seasonality), calendar indicators, `beta_0`.

Window: starts at `gas_regional.hourly_valid_start` (2023-06-01) — before
that date the regional intraday profile is phase-shifted (see
src/regional_gas.py), unsuitable for an hourly model.

Output: data/processed/regional/dataset_region_<code>.parquet (one per
region) + qc_regional.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.build_dataset import (
    load_holidays,
    load_school_holidays,
    run_qc,
)
from src.calendar_features import (
    compute_end_of_year_flag,
    compute_off_peak_flag,
    compute_weekday_flags,
)
from src.config import load_config, resolve_path
from src.fourier_features import compute_fourier_features
from src.regional_gas import load_regional_gas
from src.regional_meteo import load_meteo_regional
from src.thermal_features import (
    compute_temp_smo,
    compute_x1_heating,
    compute_x2_smo_heating,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_regional_dataset")


def _build_shared_features(config: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """National features (identical for every region) over the UTC index."""
    fourier_cfg = config["fourier"]
    calendar_tz = config["timezone"]["calendar_reference"]

    fourier_df = compute_fourier_features(
        index, fourier_cfg["harmonics"], fourier_cfg["days_in_year"], calendar_tz
    )
    weekday_flags = compute_weekday_flags(index, calendar_tz)
    end_of_year = compute_end_of_year_flag(index, calendar_tz, config["calendar"]["end_of_year_window"])
    holiday_dates = load_holidays(config)
    school_dates = load_school_holidays(config)
    off_peak = compute_off_peak_flag(index, calendar_tz, holiday_dates, school_dates, end_of_year)
    beta_0 = pd.Series(1, index=index, name="beta_0")

    return pd.concat([fourier_df, weekday_flags, end_of_year, off_peak, beta_0], axis=1)


def build_regional(config: dict) -> tuple[dict[int, pd.DataFrame], dict]:
    gas_r = load_regional_gas(config)       # UTC x code_region
    temp_r = load_meteo_regional(config)    # UTC x code_region

    # Window: reliable hourly history only.
    start = pd.Timestamp(config["gas_regional"]["hourly_valid_start"])
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    end_cfg = config["date_range"]["end"]
    end = pd.Timestamp(end_cfg) if end_cfg else min(gas_r.index.max(), temp_r.index.max())
    index = pd.date_range(start, end, freq="h", tz="UTC")
    logger.info("regional master index: %s -> %s (%d hourly points)", index.min(), index.max(), len(index))

    shared = _build_shared_features(config, index)
    thermal_cfg = config["thermal"]
    names = {int(k): v for k, v in config["gas_regional"]["regions"].items()}

    regions = sorted(set(gas_r.columns) & set(temp_r.columns) & set(names))
    frames: dict[int, pd.DataFrame] = {}
    qc: dict = {"hourly_valid_start": str(start), "regions": {}}

    for code in regions:
        y = gas_r[code].reindex(index).rename("y_gas_mw")
        temp = temp_r[code].reindex(index).rename("temp_raw_c")
        x1 = compute_x1_heating(temp, thermal_cfg["t_base_celsius"])
        temp_smo = compute_temp_smo(temp, thermal_cfg["kappa"])
        x2 = compute_x2_smo_heating(temp_smo, thermal_cfg["t_base_celsius"])

        frame = pd.concat([y, temp, x1, temp_smo, x2, shared], axis=1)
        frames[code] = frame

        region_qc = run_qc(frame, index)
        region_qc["region_label"] = names[code]
        qc["regions"][str(code)] = region_qc
        logger.info(
            "region %s (%s): %d rows, y NaN=%.2f%%, temp NaN=%.2f%%",
            code, names[code], len(frame),
            100 * frame["y_gas_mw"].isna().mean(), 100 * frame["temp_raw_c"].isna().mean(),
        )

    return frames, qc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-subdir", default="regional", help="subdirectory of data/processed/")
    args = parser.parse_args(argv)

    config = load_config()
    frames, qc = build_regional(config)

    out_dir = resolve_path(config["output"]["processed_dir"]) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for code, frame in frames.items():
        path = out_dir / f"dataset_region_{code}.parquet"
        frame.to_parquet(path)
    logger.info("wrote %d per-region parquet files to %s", len(frames), out_dir)

    qc_path = out_dir / "qc_regional.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote regional QC report to %s", qc_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
