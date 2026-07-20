"""Benchmark 1: 24 independent OLS equations (one per local hour).

Each hour h has its own coefficients, estimated separately with
`statsmodels.OLS`, sharing no information across equations. This is the naive
baseline against which the SURE system's gain (or lack thereof) is measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.dataset import PREDICTOR_COLUMNS, TARGET_COLUMN


class HourlyOLSModel:
    def __init__(self, predictor_cols: list[str] = PREDICTOR_COLUMNS):
        self.predictor_cols = list(predictor_cols)
        self.results_: dict[int, sm.regression.linear_model.RegressionResultsWrapper] = {}
        # Derived statistics/coefficients, extracted at fit time rather than
        # read later from `results_`: `models.persistence` strips the raw data
        # (exog/endog) from the results before saving to keep artifacts small,
        # which breaks lazily computed properties (mse_resid, rsquared) AND
        # loses the column names of `.params` after a pickle round-trip (it
        # falls back to a positional integer index) — verified empirically.
        # `beta_` is therefore the only source used for prediction, never
        # `results_[h].params` directly.
        self.beta_: dict[int, np.ndarray] = {}
        self.mse_resid_: dict[int, float] = {}
        self.rsquared_: dict[int, float] = {}

    def fit(self, train_per_hour: dict[int, pd.DataFrame], target_col: str = TARGET_COLUMN) -> "HourlyOLSModel":
        for h, frame in train_per_hour.items():
            y = frame[target_col]
            X = frame[self.predictor_cols]
            res = sm.OLS(y, X).fit()
            self.results_[h] = res
            self.beta_[h] = res.params[self.predictor_cols].to_numpy(dtype=float)
            self.mse_resid_[h] = float(res.mse_resid)
            self.rsquared_[h] = float(res.rsquared)
        return self

    def predict(self, per_hour: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
        preds = {}
        for h, frame in per_hour.items():
            x = frame[self.predictor_cols].to_numpy(dtype=float)
            pred = x @ self.beta_[h]
            preds[h] = pd.Series(pred, index=frame["utc_ts"].to_numpy(), name=f"ols_pred_h{h}")
        return preds

    def coefficients(self) -> pd.DataFrame:
        """Coefficients (hours x predictors), for inspection."""
        return pd.DataFrame(self.beta_, index=self.predictor_cols).T.sort_index()

    def r_squared_by_hour(self) -> pd.Series:
        return pd.Series(self.rsquared_).sort_index()
