"""Evaluation metrics: RMSE, MAPE, MAE, plus per-hour/per-model aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.sqrt(np.mean((y_true.to_numpy() - y_pred.to_numpy()) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.mean(np.abs((y_true.to_numpy() - y_pred.to_numpy()) / y_true.to_numpy())) * 100)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n": int(len(y_true)),
    }


def combine_hourly(preds: dict[int, pd.Series]) -> pd.Series:
    """Concatenate the 24 hourly series (indexed by `utc_ts`) into one sorted series."""
    return pd.concat(preds.values()).sort_index()


def evaluate_hourly(per_hour_true: dict[int, pd.Series], per_hour_pred: dict[int, pd.Series]) -> pd.DataFrame:
    rows = [{"hour": h, **evaluate(per_hour_true[h], per_hour_pred[h])} for h in sorted(per_hour_true)]
    return pd.DataFrame(rows).set_index("hour")


def evaluate_overall(per_hour_true: dict[int, pd.Series], per_hour_pred: dict[int, pd.Series]) -> dict:
    return evaluate(combine_hourly(per_hour_true), combine_hourly(per_hour_pred))
