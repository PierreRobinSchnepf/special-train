import numpy as np
import pandas as pd
import pytest

from src.calendar_features import (
    compute_end_of_year_flag,
    compute_off_peak_flag,
    compute_weekday_flags,
)
from src.fourier_features import compute_fourier_features
from src.thermal_features import compute_temp_smo, compute_x1_heating, compute_x2_smo_heating


# ---------------------------------------------------------------------------
# Thermal block
# ---------------------------------------------------------------------------

def test_x1_heating_clips_at_zero():
    temp = pd.Series([20.0, 15.0, 10.0, -5.0])
    x1 = compute_x1_heating(temp, t_base=15.0)
    assert list(x1) == [0.0, 0.0, 5.0, 20.0]


def test_temp_smo_recursion_matches_manual_formula():
    temp = pd.Series([10.0, 0.0, 20.0, 5.0, -3.0])
    kappa = 0.98
    smo = compute_temp_smo(temp, kappa)

    expected = [temp.iloc[0]]
    for t in temp.iloc[1:]:
        expected.append(kappa * expected[-1] + (1 - kappa) * t)

    np.testing.assert_allclose(smo.to_numpy(), expected)


def test_temp_smo_initializes_at_first_observation():
    temp = pd.Series([7.5, 10.0, 12.0])
    smo = compute_temp_smo(temp, kappa=0.98)
    assert smo.iloc[0] == pytest.approx(7.5)


def test_x2_uses_smoothed_not_raw_temperature():
    temp = pd.Series([20.0, -10.0])  # sudden cold snap
    smo = compute_temp_smo(temp, kappa=0.98)
    x2 = compute_x2_smo_heating(smo, t_base=15.0)
    # kappa=0.98 barely reacts to a single-step shock -> X2 stays ~0 at t=1
    assert x2.iloc[1] < 1.0


def test_x1_reacts_immediately_to_cold_snap():
    temp = pd.Series([20.0, -10.0])
    x1 = compute_x1_heating(temp, t_base=15.0)
    assert x1.iloc[1] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Fourier block
# ---------------------------------------------------------------------------

def test_fourier_masks_weekday_vs_weekend():
    # 2024-01-01 is a Monday, 2024-01-06 is a Saturday
    index = pd.DatetimeIndex(
        ["2024-01-01T12:00:00Z", "2024-01-06T12:00:00Z"], tz="UTC"
    )
    df = compute_fourier_features(index, harmonics=[1, 2, 3, 4], days_in_year=365.25, calendar_tz="Europe/Paris")

    monday, saturday = df.iloc[0], df.iloc[1]
    assert monday["cos1_WD"] != 0.0
    assert monday["cos1_WE"] == 0.0
    assert saturday["cos1_WE"] != 0.0
    assert saturday["cos1_WD"] == 0.0


def test_fourier_values_bounded():
    index = pd.date_range("2024-01-01", "2024-12-31", freq="D", tz="UTC")
    df = compute_fourier_features(index, harmonics=[1, 2, 3, 4], days_in_year=365.25, calendar_tz="Europe/Paris")
    assert (df.to_numpy() >= -1.0).all()
    assert (df.to_numpy() <= 1.0).all()


def test_fourier_utc_late_hours_use_next_paris_day():
    # 23:30 UTC in winter = 00:30 Paris the next calendar day.
    idx_utc_evening = pd.DatetimeIndex(["2024-01-01T23:30:00Z"], tz="UTC")
    idx_utc_next_morning = pd.DatetimeIndex(["2024-01-02T06:00:00Z"], tz="UTC")

    df_evening = compute_fourier_features(idx_utc_evening, [1], 365.25, "Europe/Paris")
    df_morning = compute_fourier_features(idx_utc_next_morning, [1], 365.25, "Europe/Paris")

    np.testing.assert_allclose(df_evening.to_numpy(), df_morning.to_numpy())


# ---------------------------------------------------------------------------
# Calendar block
# ---------------------------------------------------------------------------

def test_weekday_flags_are_mutually_exclusive_on_named_days():
    index = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC")  # Mon..Sun
    flags = compute_weekday_flags(index, "Europe/Paris")
    assert flags["is_monday"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert flags["is_friday"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert flags["is_saturday"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert flags["is_sunday"].tolist() == [0, 0, 0, 0, 0, 0, 1]


def test_end_of_year_window():
    index = pd.DatetimeIndex(
        ["2024-12-23T12:00:00Z", "2024-12-24T12:00:00Z", "2024-12-31T12:00:00Z", "2025-01-01T12:00:00Z"],
        tz="UTC",
    )
    flag = compute_end_of_year_flag(index, "Europe/Paris", {"start_month": 12, "start_day": 24, "end_month": 12, "end_day": 31})
    assert flag.tolist() == [0, 1, 1, 0]


def test_off_peak_combines_holiday_school_and_end_of_year():
    index = pd.DatetimeIndex(
        [
            "2024-07-15T12:00:00Z",  # ordinary Monday
            "2024-05-01T12:00:00Z",  # May 1st, public holiday
            "2024-12-26T12:00:00Z",  # end of year window
        ],
        tz="UTC",
    )
    end_of_year = compute_end_of_year_flag(index, "Europe/Paris", {"start_month": 12, "start_day": 24, "end_month": 12, "end_day": 31})
    holidays = {pd.Timestamp("2024-05-01").date()}
    school = set()
    off_peak = compute_off_peak_flag(index, "Europe/Paris", holidays, school, end_of_year)
    assert off_peak.tolist() == [0, 1, 1]


def test_off_peak_school_holiday_triggers_flag():
    index = pd.DatetimeIndex(["2024-02-15T12:00:00Z"], tz="UTC")
    end_of_year = pd.Series([0], index=index)
    school = {pd.Timestamp("2024-02-15").date()}
    off_peak = compute_off_peak_flag(index, "Europe/Paris", set(), school, end_of_year)
    assert off_peak.tolist() == [1]
