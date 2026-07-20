"""Unit tests of the regional gas parsing (wide local time -> long UTC).

Pure (no network): synthetic wide dataframes reproduce the observed quirks
of the two ODRÉ datasets (inconsistent hourly column names, several rows per
region for operators and sectors) and the aggregation and timezone
conversion are verified.
"""
import numpy as np
import pandas as pd

from scripts.build_dataset import renormalized_weighted_mean
from src.regional_gas import _hour_columns, _mask_isolated_dips, _wide_to_regional_utc


def test_hour_columns_tolerates_naming_variants():
    df = pd.DataFrame(
        columns=["date", "code_region", "consommation_journaliere_mwh_pcs",
                 "00_00_00", "06_00", "11_00_00", "23_00"]
    )
    mapping = _hour_columns(df)
    # hourly columns recognized whatever the suffix, non-hourly ones ignored
    assert mapping == {"00_00_00": 0, "06_00": 6, "11_00_00": 11, "23_00": 23}


def test_sums_over_operators_and_sectors_per_region():
    # two operators x two sectors for the same region/day: must be summed
    df = pd.DataFrame(
        {
            "date": ["2025-12-01"] * 4,
            "code_region": [11, 11, 11, 11],
            "operateur": ["NaTran", "NaTran", "Terega", "Terega"],
            "secteur": ["A", "B", "A", "B"],
            "12_00": [10.0, 20.0, 1.0, 4.0],
        }
    )
    out = _wide_to_regional_utc(df, "code_region")
    # 12:00 Paris local time in winter (CET=UTC+1) -> 11:00 UTC
    ts = pd.Timestamp("2025-12-01 11:00", tz="UTC")
    assert list(out.columns) == [11]
    assert out.loc[ts, 11] == 35.0


def test_local_to_utc_conversion_winter():
    df = pd.DataFrame(
        {"date": ["2025-12-01"], "code_region": [24], "00_00": [50.0]}
    )
    out = _wide_to_regional_utc(df, "code_region")
    # midnight Paris local time in winter -> 23:00 UTC the previous day
    assert out.index[0] == pd.Timestamp("2025-11-30 23:00", tz="UTC")
    assert out.iloc[0, 0] == 50.0


def test_regions_become_separate_columns():
    df = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-01"],
            "code_region": [11, 75],
            "08_00": [100.0, 200.0],
        }
    )
    out = _wide_to_regional_utc(df, "code_region")
    assert set(out.columns) == {11, 75}
    ts = pd.Timestamp("2025-12-01 07:00", tz="UTC")
    assert out.loc[ts, 11] == 100.0
    assert out.loc[ts, 75] == 200.0


# ---------------------------------------------------------------------------
# Shared national/regional weighting (Step B)
# ---------------------------------------------------------------------------

def test_weighted_mean_matches_manual():
    wide = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 40.0]})
    out = renormalized_weighted_mean(wide, {"a": 3.0, "b": 1.0})
    # row 0: (10*3 + 0*1)/4 = 7.5; row 1: (20*3 + 40*1)/4 = 25
    np.testing.assert_allclose(out.to_numpy(), [7.5, 25.0])


def test_weighted_mean_renormalizes_on_missing():
    # when a station is missing, the weights renormalize over the available ones
    wide = pd.DataFrame({"a": [10.0, np.nan], "b": [20.0, 30.0]})
    out = renormalized_weighted_mean(wide, {"a": 1.0, "b": 1.0})
    # row 0: (10+20)/2 = 15; row 1: b alone -> 30 (not (0+30)/2)
    np.testing.assert_allclose(out.to_numpy(), [15.0, 30.0])


def test_weighted_mean_all_missing_is_nan():
    wide = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    out = renormalized_weighted_mean(wide, {"a": 1.0, "b": 2.0})
    assert np.isnan(out.iloc[0])


# ---------------------------------------------------------------------------
# Isolated-dip filter (DST artefact of the regional source)
# ---------------------------------------------------------------------------

def test_isolated_dip_masked_but_smooth_trough_kept():
    # column A: isolated one-hour collapse (1000 -> 5 -> 1000) => masked
    # column B: smooth summer trough (130 -> 104 -> 110) => kept
    df = pd.DataFrame({
        "A": [1000.0, 5.0, 1000.0],
        "B": [130.0, 104.0, 110.0],
    })
    out = _mask_isolated_dips(df, dip_fraction=0.5)
    assert np.isnan(out.loc[1, "A"])       # anomaly masked
    assert out.loc[1, "B"] == 104.0         # true trough intact
    assert out.loc[0, "A"] == 1000.0        # neighbors intact


def test_isolated_dip_edge_values_never_masked():
    # an edge value (missing one of its two neighbors) is never masked
    df = pd.DataFrame({"A": [1.0, 1000.0, 1000.0]})
    out = _mask_isolated_dips(df, dip_fraction=0.5)
    assert out.loc[0, "A"] == 1.0
