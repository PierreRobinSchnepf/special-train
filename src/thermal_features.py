"""Thermal block of Table 1: X1_heating, temp_smo, X2_smo_heating.

Every function is causal: it only consumes the past and present of the input
series (no data leakage).
"""
from __future__ import annotations

import pandas as pd


def compute_x1_heating(temp: pd.Series, t_base: float) -> pd.Series:
    """X1_t = max(0, t_base - T_t) — immediate reaction to cold."""
    return (t_base - temp).clip(lower=0.0).rename("X1_heating")


def compute_temp_smo(temp: pd.Series, kappa: float) -> pd.Series:
    """T_smo_t = kappa * T_smo_{t-1} + (1-kappa) * T_t.

    Equivalent to a causal EWMA with smoothing factor alpha = 1 - kappa and
    adjust=False (exact recursion, no renormalization by the weights of past
    observations). T_smo_0 = T_0 (the "first_observation" choice documented in
    config.yaml, matching the native behavior of pandas.ewm(adjust=False)).

    Internal NaNs in `temp` must be filled (bounded ffill) by the caller before
    calling: pandas.ewm silently carries the last non-null value through NaNs,
    which would mask undocumented data gaps.
    """
    alpha = 1.0 - kappa
    return temp.ewm(alpha=alpha, adjust=False).mean().rename("temp_smo")


def compute_x2_smo_heating(temp_smo: pd.Series, t_base: float) -> pd.Series:
    """X2_t = max(0, t_base - T_smo_t) — heating inertia."""
    return (t_base - temp_smo).clip(lower=0.0).rename("X2_smo_heating")
