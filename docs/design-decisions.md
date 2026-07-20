# Design decisions

Documented, deliberate choices made while building the pipeline — including
the ones that could have been guessed silently but were not. Kept out of the
README to keep it readable; this is the reference when a choice looks
arbitrary.

## 1. Timezone: UTC as the master index

The dataset is indexed in **continuous, gap-free UTC** (`freq="h"`). This is
deliberate: Europe/Paris DST transitions (duplicated hour in late October,
nonexistent hour in late March) do not exist in UTC, which removes the
problem structurally instead of handling it case by case. Variables that
depend on the French civil calendar (weekday, day of year for Fourier,
public holidays, school breaks, end-of-year window) are computed by
converting the UTC timestamp to `Europe/Paris` **only to extract the local
date**, never by re-indexing on it. See `src/calendar_features.py` and
`src/fourier_features.py`.

Verified empirically (2026-07-16):
- ODRÉ `date_heure` is already UTC (23:30 Paris local time ↔ 21:30Z in
  summer, +2h offset = CEST — consistent).
- Météo-France `AAAAMMJJHH` follows the standard SYNOP convention (UTC).

## 2. ODRÉ endpoint choice

The field requested in the initial spec (`gaz_grtgaz`) no longer exists: the
schema evolved into an aggregated `consommation_brute_gaz_totale` field
(GRTgaz + Teréga, continental France). The v1 endpoint
`/api/records/1.0/download/` **silently truncates around 4 MB** (verified:
cut in the middle of a CSV row) — unusable for 8 years of history. The
pipeline therefore uses the v2 `/exports/csv` endpoint with a
`where=date_heure in [date'YYYY-01-01'..date'YYYY-12-31']` filter, chunked
by civil year, which has neither the offset+limit≤10000 cap of `/records`
nor the size truncation of `/download/`.

## 3. Weather-station selection and national weighting

The report asks for a weighting "by population/gas consumption per zone"
without providing a public GRTgaz zoning. Documented choice (not silently
guessed): a basket of 16 departments representative of the major climatic/
demographic zones of continental France, weighted by an order of magnitude
of department population (INSEE ~2021, `config.yaml § meteo.stations`). For
each department, the weather station actually used is chosen **at build
time**, automatically, as the one with the best non-null coverage of `T` —
logged to `data/raw/meteo/station_selection.json` on every run, not frozen
in the code.

## 4. Table 1 ambiguities — flagged, not silently resolved

- **Mid-August dip**: mentioned in the report's text but absent from
  Table 1. **Not implemented.** No `is_mid_august` column.
- **`is_off_peak_period`**: no single operational definition in the report.
  **Proposed and documented** definition:
  `holiday OR any_school_zone_on_break OR is_end_of_year`
  (see `docs/data-dictionary.md` for the rationale and the deliberate
  overlap with `is_end_of_year`).

## 5. Causality of the features

- `X1_heating` depends only on `T_t` (present).
- `temp_smo` / `X2_smo_heating` are an EWMA recursion (`κ=0.98`) that only
  looks at the past and the present (`temp_smo_0 = T_0`, no backfill from
  the future).
- Fourier and calendar variables depend only on the current timestamp's
  date.
- No smoothing or accumulation in the pipeline ever "sees" a future
  observation.

## 6. Statsmodels artifacts and `remove_data()`

Statsmodels results embed their full design matrix by default (hundreds of
MB once pickled across 24 hours). `models/persistence.py` calls
`remove_data()` on them before saving to shrink artifacts (~50x), but that
breaks `.params` (loses column names after a pickle round-trip) and lazily
computed properties (`.mse_resid`, `.rsquared`) not already cached. Hence
`HourlyOLSModel.beta_`/`.mse_resid_`/`.rsquared_` and
`HourlySUREModel.stage1_resid_var_`: everything needed after loading is
extracted into plain attributes (numpy/dict) **during** `fit()`, never read
back from the statsmodels object afterwards.

## 7. Regional data constraints

The national target `consommation-quotidienne-brute` has no regional
breakdown, so regional consumption is reconstructed by summing two regional
ODRÉ datasets (industrial + public distribution) per region. Sum of the 12
regions ≈ national total to ~0.03% (2024+).

Documented data constraint (`src/regional_gas.py`): before **June 2023**,
the intraday hourly profile of these datasets is phase-shifted (~+6h;
distribution peak at 12h UTC in 2018-2022 vs 06h UTC in 2024+), while daily
totals stay correct. Hourly regional models therefore only train on the
clean history `>= gas_regional.hourly_valid_start` (2023-06-01,
`config.yaml`). A recurring artefact on the Saturday 01:00 UTC preceding
the March DST switch (aberrant values 0→~100 MW) is filtered by an
**isolated-dip** criterion (`_mask_isolated_dips`: an hour < 0.5× its
neighbors' minimum is an impossible collapse given thermal inertia) rather
than a magnitude threshold (the true summer minimum grazes the anomaly).

## 8. Data freshness and the live pipeline

Freshness research conducted before implementing the live pipeline:

| Source | Scope | Observed lag |
|---|---|---|
| `consommation-quotidienne-brute` (training target) | National, total | **~45-50 days** |
| industrial + distribution regional datasets | Regional, 2 operators | **~15-20 days** |
| Météo-France / Open-Meteo | National (16 stations) | near real time |

The two regional datasets, summed over all regions and both operators
(NaTran + Teréga), reconstruct the national target to within ~0.2%
(verified empirically) — the best achievable freshness for the full target.
Since weather stays near-real-time while gas caps at ~15-20 days of lag,
the pipeline decouples the two:

- The **Kalman state** is only re-assimilated from available ground truth,
  up to **day G** (~15-20 days back).
- The **delivered forecast stays J (17:00) → J+1 (23:00)**: the state
  frozen at day G is propagated (random walk, no update) through **real**
  weather to fill day G → today, then the **Open-Meteo** weather forecast
  (Météo-France AROME model, free, keyless — the official Météo-France API
  requires a manual registration an agent cannot perform for the user) only
  for today → J+1.

## 9. Why Open-Meteo and not the official Météo-France API

The official API (portail-api.meteofrance.fr) requires manual registration
(email + validation). Open-Meteo is free, keyless, and serves precisely the
Météo-France AROME model forecasts for France (`/v1/meteofrance`) — tested
and validated for this project. Stations and weights are identical to
training; only the source differs.
