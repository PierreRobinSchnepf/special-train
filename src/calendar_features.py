"""Calendar block of Table 1: weekday flags, end of year, off-peak.

Every indicator is derived from the Europe/Paris civil date (see config.yaml
§ timezone), never from the raw UTC date.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def _local_dates(index: pd.DatetimeIndex, calendar_tz: str) -> pd.Series:
    if index.tz is None:
        raise ValueError("index must be tz-aware (UTC)")
    local = index.tz_convert(calendar_tz)
    return pd.Series(local.date, index=index)


def compute_weekday_flags(index: pd.DatetimeIndex, calendar_tz: str) -> pd.DataFrame:
    local = index.tz_convert(calendar_tz)
    dow = local.dayofweek  # 0=Monday ... 6=Sunday
    return pd.DataFrame(
        {
            "is_monday": (dow == 0).astype("int8"),
            "is_friday": (dow == 4).astype("int8"),
            "is_saturday": (dow == 5).astype("int8"),
            "is_sunday": (dow == 6).astype("int8"),
        },
        index=index,
    )


def compute_end_of_year_flag(
    index: pd.DatetimeIndex,
    calendar_tz: str,
    window: dict,
) -> pd.Series:
    local = index.tz_convert(calendar_tz)
    start_month, start_day = window["start_month"], window["start_day"]
    end_month, end_day = window["end_month"], window["end_day"]

    if start_month == end_month:
        # Window contained in a single month (current config: Dec 24-31).
        in_window = (local.month == start_month) & (local.day >= start_day) & (local.day <= end_day)
    else:
        # Window straddling the new year (e.g. Dec 24 -> Jan 5).
        in_window = ((local.month == start_month) & (local.day >= start_day)) | (
            (local.month == end_month) & (local.day <= end_day)
        )
    return pd.Series(in_window.astype("int8"), index=index, name="is_end_of_year")


def compute_off_peak_flag(
    index: pd.DatetimeIndex,
    calendar_tz: str,
    holiday_dates: set[date],
    school_holiday_dates: set[date],
    is_end_of_year: pd.Series,
) -> pd.Series:
    """is_off_peak_period = holiday OR any-zone school break OR end_of_year.

    Proposed, documented definition (the source report does not give a single
    operational definition) — see config.yaml § calendar.off_peak_definition
    and docs/data-dictionary.md.
    """
    local_dates = _local_dates(index, calendar_tz)
    is_holiday = local_dates.isin(holiday_dates)
    is_school_break = local_dates.isin(school_holiday_dates)
    combined = is_holiday | is_school_break | is_end_of_year.astype(bool)
    return combined.astype("int8").rename("is_off_peak_period")
