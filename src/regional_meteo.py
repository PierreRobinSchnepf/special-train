"""PER-REGION temperature (Step B of the region-by-region model).

Reuses exactly the same machinery as the national temperature
(`build_dataset.load_per_department_temp` + `renormalized_weighted_mean`):
automatic selection of the best-covered station per department, short-gap
filling, then population-weighted average. The only difference is the
grouping: instead of aggregating the 16 departments into a single national
series, they are grouped by `region_code` (INSEE) to produce one temperature
per region, aligned with the 12 gas regions.

Every gas region is covered by >= 1 department of the weather basket (see
config.yaml § meteo.stations, `region_code`). Departments whose `region_code`
matches none of the 12 gas regions are ignored (there are none today, but the
function stays robust if the basket evolves).
"""
from __future__ import annotations

import logging

import pandas as pd

from scripts.build_dataset import load_per_department_temp, renormalized_weighted_mean

logger = logging.getLogger(__name__)


def load_meteo_regional(config: dict) -> pd.DataFrame:
    """Hourly temperature per region, UTC-indexed. Returns a DataFrame
    (UTC index) x (code_region int), in °C. Each column is the population-
    weighted average of the region's stations, with weights renormalized over
    the stations available at each hour.
    """
    meteo_cfg = config["meteo"]
    per_dept_series, weights = load_per_department_temp(config)

    # dep -> region_code, from config (single source of truth).
    dep_to_region: dict[str, int] = {}
    for station_cfg in meteo_cfg["stations"]:
        code = station_cfg.get("region_code")
        if code is not None:
            dep_to_region[station_cfg["department"]] = int(code)

    # Expected gas regions (the regional target only covers these).
    gas_regions = {int(k) for k in config["gas_regional"]["regions"]}

    wide = pd.DataFrame(per_dept_series)
    max_gap = meteo_cfg["max_ffill_gap_hours"]
    wide_filled = wide.ffill(limit=max_gap)

    region_series: dict[int, pd.Series] = {}
    for region_code in sorted(gas_regions):
        deps = [d for d, r in dep_to_region.items() if r == region_code and d in wide_filled.columns]
        if not deps:
            logger.warning("region %s has no meteo station, temperature will be all-NaN", region_code)
            continue
        sub = wide_filled[deps]
        sub_weights = {d: weights[d] for d in deps}
        region_series[region_code] = renormalized_weighted_mean(sub, sub_weights)
        logger.info(
            "region %s: temperature from %d station(s) %s", region_code, len(deps), deps
        )

    regional_temp = pd.DataFrame(region_series)
    missing = gas_regions - set(regional_temp.columns)
    if missing:
        logger.warning("regions without any temperature series: %s", sorted(missing))
    return regional_temp
