# Data dictionary — `dataset_final.parquet`

Index: hourly, UTC, continuous, gap-free (`pandas.date_range(freq="h", tz="UTC")`).
One row = one UTC hour. Calendar/seasonal variables are computed on the
**Europe/Paris** civil date derived from this index (see `config.yaml §
timezone` and the README for the rationale).

| Column | Unit | Source | Formula / definition |
|---|---|---|---|
| `y_gas_mw` | MW PCS 0°C | ODRÉ `consommation-quotidienne-brute`, field `consommation_brute_gaz_totale` (GRTgaz+Terega, continental France) | Raw hourly value, forecasting target. |
| `temp_raw_c` | °C | Météo-France, hourly data by department, weighted national average | Weighted average (weights = approximate department population, INSEE, see `config.yaml § meteo.stations`) of the `T` readings of the selected stations, one station per department chosen automatically (best non-null coverage over the period). |
| `X1_heating` | °C (deficit) | Derived from `temp_raw_c` | `max(0, T_base - T_t)`, `T_base = 15°C`. Immediate reaction to cold. |
| `temp_smo` | °C | Derived from `temp_raw_c` | `T_smo_t = κ·T_smo_{t-1} + (1-κ)·T_t`, `κ = 0.98`, `T_smo_0 = T_0`. Global thermal inertia (Dordonnat et al.). |
| `X2_smo_heating` | °C (deficit) | Derived from `temp_smo` | `max(0, T_base - T_smo_t)`. Heating inertia. |
| `cos1_WD`…`cos4_WD`, `sin1_WD`…`sin4_WD` | unitless, [-1,1] | Calendar (generated) | `cos(2π·s·d/365.25)` / `sin(...)`, `d` = day of year (Europe/Paris), value kept on weekdays, **0 otherwise**. |
| `cos1_WE`…`cos4_WE`, `sin1_WE`…`sin4_WE` | unitless, [-1,1] | Calendar (generated) | Same, value kept on weekends, **0 otherwise**. |
| `is_monday` | {0,1} | Calendar (generated) | Europe/Paris civil day = Monday. |
| `is_friday` | {0,1} | Calendar (generated) | Europe/Paris civil day = Friday. |
| `is_saturday` | {0,1} | Calendar (generated) | Europe/Paris civil day = Saturday. |
| `is_sunday` | {0,1} | Calendar (generated) | Europe/Paris civil day = Sunday. |
| `is_end_of_year` | {0,1} | Calendar (generated) | Europe/Paris civil day ∈ [Dec 24, Dec 31] (configurable window, `config.yaml § calendar.end_of_year_window`). |
| `is_off_peak_period` | {0,1} | Public holidays (etalab) + school holidays (data.education.gouv.fr) + `is_end_of_year` | **Proposed definition, not fixed by the source report (flagged ambiguity):** `holiday OR any_school_zone_on_break OR is_end_of_year`. Metropolitan school zones A/B/C only. See README § Ambiguities. |
| `beta_0` | constant | Generated | Always 1. Base load / incompressible residual consumption in the model. |

## Intermediate columns excluded from Table 1 but kept for audit

None — `temp_raw_c` is the only additional "raw" variable kept beyond
Table 1, because `X1_heating` and `X2_smo_heating` derive directly from it
and its presence allows both formulas to be verified after the fact.

## National temperature: aggregation method

The source report asks for a weighting "by population/gas consumption per
zone" without providing an official GRTgaz zoning. Documented choice:
weighting by department population (INSEE ~2021 order of magnitude) of the
16 departments in the basket (`config.yaml § meteo.stations`), covering the
main climatic and demographic zones of continental France. If a department
is missing at a given hour (station offline beyond the
`max_ffill_gap_hours` fill tolerance), the weights of the remaining
departments are renormalized for that hour (logged in QC).

## Regional datasets — `dataset_region_<code>.parquet`

Same schema as the national dataset, one file per gas region (12 regions,
INSEE codes in `config.yaml § gas_regional.regions`). Differences:

- `y_gas_mw` is the region's consumption, reconstructed as industrial +
  public distribution (two regional ODRÉ datasets, summed per region);
- `temp_raw_c` and the thermal block use the region's population-weighted
  station average;
- the window starts at `gas_regional.hourly_valid_start` (2023-06-01),
  because the regional intraday profile is phase-shifted before that date
  (see `src/regional_gas.py`).

## Flagged ambiguities (not silently resolved)

1. **Mid-August dip**: the source report's text mentions a possible
   mid-August consumption dip (summer industrial closures), not reflected
   in Table 1. No column was added for this effect — to be confirmed before
   implementation.
2. **`is_off_peak_period`**: Table 1 gives no unambiguous operational
   definition ("school holidays, public holidays and end-of-year break").
   The definition above is an explicit, documented choice, with deliberate
   overlap with `is_end_of_year` (both columns are kept separate so a model
   can estimate a differential effect).

## Weather-station selection traceability

The station kept per department (among all stations present in the
Météo-France department file) is chosen automatically at build time as the
one with the most non-null `T` observations over the period. The per-
department details (selected station, observation count, temporal coverage)
are written to `data/raw/meteo/station_selection.json` on every
`scripts/build_dataset.py` run, for audit.
